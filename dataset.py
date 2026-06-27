import copy
import random

from utils import *
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def _sample_noise_count(length, ratio, rng):
    expected = length * ratio
    count = int(expected)
    if rng.random() < expected - count:
        count += 1
    return count


def _insert_aligned_time(time_session, position):
    if len(time_session) == 0:
        return 0
    if position <= 0:
        return time_session[0]
    if position >= len(time_session):
        return time_session[-1]
    return time_session[position - 1]


def inject_random_noisy_pois(
    sessions_dict,
    num_pois,
    noise_ratio,
    seed=None,
    labels_dict=None,
    time_sessions_dict=None,
):
    if noise_ratio < 0 or noise_ratio > 1:
        raise ValueError("noise_ratio must be in [0, 1], got {}".format(noise_ratio))

    noisy_sessions = copy.deepcopy(sessions_dict)
    noisy_time_sessions = (
        copy.deepcopy(time_sessions_dict) if time_sessions_dict is not None else None
    )
    stats = {
        "noise_ratio": float(noise_ratio),
        "trajectories": 0,
        "changed_trajectories": 0,
        "inserted_pois": 0,
    }
    if noise_ratio == 0:
        return noisy_sessions, noisy_time_sessions, stats

    rng = random.Random(seed)
    all_pois = list(range(num_pois))

    for user_id, sessions in noisy_sessions.items():
        for session_idx, session in enumerate(sessions):
            stats["trajectories"] += 1
            original_session = list(session)
            noisy_session = list(session)
            noisy_sessions[user_id][session_idx] = noisy_session

            if len(original_session) == 0:
                continue

            noise_count = _sample_noise_count(len(original_session), noise_ratio, rng)
            if noise_count <= 0:
                continue

            excluded_pois = set(original_session)
            if labels_dict is not None and user_id in labels_dict:
                excluded_pois.add(labels_dict[user_id])
            candidate_pois = [poi for poi in all_pois if poi not in excluded_pois]
            if not candidate_pois:
                continue

            noise_pois = rng.sample(
                candidate_pois, min(noise_count, len(candidate_pois))
            )
            noisy_time_session = None
            if noisy_time_sessions is not None:
                noisy_time_session = list(noisy_time_sessions[user_id][session_idx])
                noisy_time_sessions[user_id][session_idx] = noisy_time_session

            for noise_poi in noise_pois:
                insert_pos = rng.randint(0, len(noisy_session))
                noisy_session.insert(insert_pos, noise_poi)
                if noisy_time_session is not None:
                    noisy_time_session.insert(
                        insert_pos, _insert_aligned_time(noisy_time_session, insert_pos)
                    )

            stats["inserted_pois"] += len(noise_pois)
            stats["changed_trajectories"] += 1

    return noisy_sessions, noisy_time_sessions, stats


def select_adaptive_poi_changes(
    user_traj,
    similarity_matrix,
    add_threshold=0.1,
    remove_threshold=0.0,
    update_ratio=0.1,
):
    original_pois = set(int(poi) for poi in user_traj)
    original_length = len(user_traj)
    select_k = int(original_length * update_ratio)
    if original_length == 0 or select_k == 0:
        return {
            "updated_pois": original_pois,
            "added_pois": set(),
            "removed_pois": set(),
            "select_k": select_k,
        }

    user_traj_tensor = torch.as_tensor(
        user_traj,
        dtype=torch.long,
        device=similarity_matrix.device,
    )

    avg_similarities = similarity_matrix[:, user_traj_tensor].mean(dim=1)
    similar_pois = torch.where(avg_similarities > add_threshold)[0]
    if len(similar_pois) > 0:
        top_k_indices = torch.argsort(
            avg_similarities[similar_pois],
            descending=True,
        )[:select_k]
        top_k_pois = set(
            int(poi) for poi in similar_pois[top_k_indices].detach().cpu().tolist()
        )
    else:
        top_k_pois = set()
    expanded_pois = original_pois.union(top_k_pois)

    poi_similarity = similarity_matrix[user_traj_tensor][:, user_traj_tensor].clone()
    poi_similarity.fill_diagonal_(-float("inf"))
    max_similarities = poi_similarity.max(dim=1)[0]
    low_similarity_positions = torch.where(max_similarities < remove_threshold)[0]
    if len(low_similarity_positions) > 0:
        bottom_k_indices = torch.argsort(
            max_similarities[low_similarity_positions],
            descending=False,
        )[:select_k]
        bottom_positions = low_similarity_positions[bottom_k_indices]
        bottom_k_pois = set(
            int(poi)
            for poi in user_traj_tensor[bottom_positions].detach().cpu().tolist()
        )
    else:
        bottom_k_pois = set()

    updated_pois = expanded_pois.difference(bottom_k_pois)
    return {
        "updated_pois": updated_pois,
        "added_pois": updated_pois.difference(original_pois),
        "removed_pois": original_pois.difference(updated_pois),
        "select_k": select_k,
    }


class POIDataset(Dataset):

    def __init__(
        self,
        data_filename,
        pois_coos_filename,
        num_users,
        num_pois,
        padding_idx,
        args,
        device,
        time_data_filename=None,
        enable_noise_poi=False,
    ):

        self.data = load_list_with_pkl(data_filename)
        self.sessions_dict = self.data[0]
        self.labels_dict = self.data[1]
        self.pois_coos_dict = load_dict_from_pkl(pois_coos_filename)

        self.time_data_filename = time_data_filename
        if self.time_data_filename is not None:
            self.time_data = load_list_with_pkl(time_data_filename)
            self.time_sessions_dict = self.time_data[0]
            self.time_labels_dict = self.time_data[1]
        else:
            self.time_data_filename = None

        self.num_users = num_users
        self.num_pois = num_pois

        self.padding_idx = padding_idx
        self.distance_threshold = args.distance_threshold
        self.keep_rate = args.keep_rate
        self.device = device
        noise_poi_ratio = getattr(args, "noise_poi_ratio", 0.0)
        noise_poi_seed = getattr(args, "noise_poi_seed", None)
        if noise_poi_seed is None:
            noise_poi_seed = getattr(args, "seed", None)
        self.noise_poi_stats = {
            "noise_ratio": float(noise_poi_ratio),
            "trajectories": 0,
            "changed_trajectories": 0,
            "inserted_pois": 0,
        }
        if enable_noise_poi and noise_poi_ratio > 0:
            self.sessions_dict, noisy_time_sessions, self.noise_poi_stats = (
                inject_random_noisy_pois(
                    sessions_dict=self.sessions_dict,
                    num_pois=self.num_pois,
                    noise_ratio=noise_poi_ratio,
                    seed=noise_poi_seed,
                    labels_dict=self.labels_dict,
                    time_sessions_dict=(
                        self.time_sessions_dict
                        if self.time_data_filename is not None
                        else None
                    ),
                )
            )
            if noisy_time_sessions is not None:
                self.time_sessions_dict = noisy_time_sessions
                self.time_data[0] = noisy_time_sessions

        self.users_trajs_dict, self.users_trajs_lens_dict = get_user_complete_traj(
            self.sessions_dict
        )
        self.users_rev_trajs_dict = get_user_reverse_traj(self.users_trajs_dict)

        if self.time_data_filename is not None:
            self.time_sessions_dict = self.time_data[0]
            self.time_labels_dict = self.time_data[1]
            self.users_time_dict, _ = get_user_complete_traj(self.time_sessions_dict)
            self.users_rev_time_dict = get_user_reverse_traj(self.users_time_dict)

        else:
            self.time_data_filename = None

        self.poi_geo_adj = gen_poi_geo_adj(
            num_pois, self.pois_coos_dict, self.distance_threshold
        )
        self.poi_geo_graph_matrix = normalized_adj(
            adj=self.poi_geo_adj, is_symmetric=False
        )
        self.poi_geo_graph = transform_csr_matrix_to_tensor(
            self.poi_geo_graph_matrix
        ).to(device)

        self.H_pu = gen_sparse_H_user(self.sessions_dict, num_pois, self.num_users)

        self.H_pu = csr_matrix_drop_edge(self.H_pu, args.keep_rate)

        self.Deg_H_pu = get_hyper_deg(self.H_pu)

        self.HG_pu = self.Deg_H_pu * self.H_pu
        self.HG_pu = transform_csr_matrix_to_tensor(self.HG_pu).to(device)

        self.H_up = self.H_pu.T
        self.Deg_H_up = get_hyper_deg(self.H_up)
        self.HG_up = self.Deg_H_up * self.H_up
        self.HG_up = transform_csr_matrix_to_tensor(self.HG_up).to(device)

        self.all_train_sessions = get_all_users_seqs(self.users_trajs_dict)

        self.pad_all_train_sessions = pad_sequence(
            self.all_train_sessions, batch_first=True, padding_value=padding_idx
        )
        self.pad_all_train_sessions = self.pad_all_train_sessions.to(device)
        self.max_session_len = self.pad_all_train_sessions.size(1)

        if args.h3_path_len:
            self.H_poi_src = gen_path_length_weighted_H_poi(
                self.users_trajs_dict, num_pois
            )
        else:
            self.H_poi_src = gen_sparse_directed_H_poi(self.users_trajs_dict, num_pois)

        self.H_poi_src = csr_matrix_drop_edge(self.H_poi_src, args.keep_rate_poi)
        self.Deg_H_poi_src = get_hyper_deg(self.H_poi_src)
        self.HG_poi_src = self.Deg_H_poi_src * self.H_poi_src
        self.HG_poi_src = transform_csr_matrix_to_tensor(self.HG_poi_src).to(device)

        self.H_poi_tar = self.H_poi_src.T
        self.Deg_H_poi_tar = get_hyper_deg(self.H_poi_tar)
        self.HG_poi_tar = self.Deg_H_poi_tar * self.H_poi_tar
        self.HG_poi_tar = transform_csr_matrix_to_tensor(self.HG_poi_tar).to(device)

        self.mg_inter_adj = gen_poi_cooccurrence_adj(self.H_pu)
        self.mg_inter_graph = transform_csr_matrix_to_tensor(
            normalized_adj(self.mg_inter_adj, is_symmetric=True)
        ).to(device)
        self.mg_dir_adj = self.H_poi_src.copy()
        self.mg_dir_graph = transform_csr_matrix_to_tensor(
            normalized_adj(self.mg_dir_adj, is_symmetric=False)
        ).to(device)
        self.mg_geo_graph = self.poi_geo_graph
        self.mg_user_poi_adj = self.H_pu.T.tocsr()
        self.mg_user_poi_graph = transform_csr_matrix_to_tensor(
            normalized_adj(self.mg_user_poi_adj, is_symmetric=False)
        ).to(device)

    def __len__(self):
        return self.num_users

    def __getitem__(self, user_idx):
        user_seq = self.users_trajs_dict[user_idx]
        user_seq_len = self.users_trajs_lens_dict[user_idx]
        user_seq_mask = [1] * user_seq_len
        user_rev_seq = self.users_rev_trajs_dict[user_idx]
        label = self.labels_dict[user_idx]

        sample = {
            "user_idx": torch.tensor(user_idx).to(self.device),
            "user_seq": torch.tensor(user_seq).to(self.device),
            "user_rev_seq": torch.tensor(user_rev_seq).to(self.device),
            "user_seq_len": torch.tensor(user_seq_len).to(self.device),
            "user_seq_mask": torch.tensor(user_seq_mask).to(self.device),
            "label": torch.tensor(label).to(self.device),
        }

        if self.time_data_filename is not None:
            user_time_seq = self.users_time_dict[user_idx]
            sample["user_time_seq"] = (
                torch.tensor(user_time_seq).float().to(self.device)
            )

        return sample

    def update_poi_session_hypergraph(
        self, similarity_matrix, add_threshold=0.1, remove_threshold=0
    ):
        copy_users_trajs_dict = copy.deepcopy(self.users_trajs_dict)

        for user_idx, user_traj in copy_users_trajs_dict.items():
            changes = select_adaptive_poi_changes(
                user_traj,
                similarity_matrix,
                add_threshold=add_threshold,
                remove_threshold=remove_threshold,
            )
            if set(user_traj) != changes["updated_pois"]:
                copy_users_trajs_dict[user_idx] = list(changes["updated_pois"])

        self.H_pu = gen_sparse_H_user_traj(
            copy_users_trajs_dict, self.num_pois, self.num_users
        )
        self.H_pu = csr_matrix_drop_edge(self.H_pu)
        self.Deg_H_pu = get_hyper_deg(self.H_pu)
        self.HG_pu = self.Deg_H_pu * self.H_pu
        self.HG_pu = transform_csr_matrix_to_tensor(self.HG_pu).to(self.device)

        self.H_up = self.H_pu.T
        self.Deg_H_up = get_hyper_deg(self.H_up)
        self.HG_up = self.Deg_H_up * self.H_up
        self.HG_up = transform_csr_matrix_to_tensor(self.HG_up).to(self.device)

    def update_user_poi_hypergraph(self, similarity_matrix_pu):

        similarity_matrix_np = similarity_matrix_pu.cpu().detach().numpy()

        binary_matrix = np.where(similarity_matrix_np > 0, 1, 0)

        new_adj = sp.csr_matrix(binary_matrix).astype(np.float64)

        self.H_pu = new_adj
        self.H_pu = csr_matrix_drop_edge(self.H_pu)
        self.Deg_H_pu = get_hyper_deg(self.H_pu)
        self.HG_pu = self.Deg_H_pu * self.H_pu
        self.HG_pu = transform_csr_matrix_to_tensor(self.HG_pu).to(self.device)

        self.H_up = self.H_pu.T
        self.Deg_H_up = get_hyper_deg(self.H_up)
        self.HG_up = self.Deg_H_up * self.H_up
        self.HG_up = transform_csr_matrix_to_tensor(self.HG_up).to(self.device)

    def update_poi_geographical_hypergraph(self, similarity_matrix):

        similarity_matrix_np = similarity_matrix.cpu().detach().numpy()

        binary_matrix = np.where(similarity_matrix_np > 0, 1, 0)

        new_adj = sp.csr_matrix(binary_matrix)
        self.poi_geo_adj = new_adj
        self.poi_geo_graph_matrix = normalized_adj(
            adj=self.poi_geo_adj, is_symmetric=False
        )
        self.poi_geo_graph = transform_csr_matrix_to_tensor(
            self.poi_geo_graph_matrix
        ).to(self.device)

    def update_poi_session_hypergraph_weights(self, similarity_matrix, threshold=0.5):

        for user_idx, user_traj in self.users_trajs_dict.items():

            for poi in user_traj:

                similar_pois = (similarity_matrix[poi] > threshold).nonzero(
                    as_tuple=True
                )[0]

                for sim_poi in similar_pois:
                    if sim_poi != poi:
                        self.H_pu[poi, user_idx] += similarity_matrix[poi, sim_poi]

        self.Deg_H_pu = get_hyper_deg(self.H_pu)
        self.HG_pu = self.Deg_H_pu * self.H_pu
        self.HG_pu = transform_csr_matrix_to_tensor(self.HG_pu).to(self.device)

    def drop_edge_HG_pu(self, keep_rate=0.9):

        self.H_pu = csr_matrix_drop_edge(self.H_pu, keep_rate)

        self.Deg_H_pu = get_hyper_deg(self.H_pu)

        self.HG_pu = self.Deg_H_pu * self.H_pu
        self.HG_pu = transform_csr_matrix_to_tensor(self.HG_pu).to(self.device)

        self.H_up = self.H_pu.T
        self.Deg_H_up = get_hyper_deg(self.H_up)
        self.HG_up = self.Deg_H_up * self.H_up
        self.HG_up = transform_csr_matrix_to_tensor(self.HG_up).to(self.device)

    def refresh_pdas_augmented_graphs(self, perturb_ratio, rng=None):
        H_pu_aug = pdas_perturb_incidence_matrix(self.H_pu, perturb_ratio, rng=rng)
        Deg_H_pu_aug = get_hyper_deg(H_pu_aug)
        self.HG_pu_aug = transform_csr_matrix_to_tensor(Deg_H_pu_aug * H_pu_aug).to(
            self.device
        )

        H_up_aug = H_pu_aug.T
        Deg_H_up_aug = get_hyper_deg(H_up_aug)
        self.HG_up_aug = transform_csr_matrix_to_tensor(Deg_H_up_aug * H_up_aug).to(
            self.device
        )

        poi_geo_adj_aug = pdas_perturb_incidence_matrix(
            self.poi_geo_adj, perturb_ratio, rng=rng
        )
        poi_geo_graph_matrix_aug = normalized_adj(
            adj=poi_geo_adj_aug, is_symmetric=False
        )
        self.poi_geo_graph_aug = transform_csr_matrix_to_tensor(
            poi_geo_graph_matrix_aug
        ).to(self.device)

        H_poi_src_aug = pdas_perturb_incidence_matrix(
            self.H_poi_src, perturb_ratio, rng=rng
        )
        Deg_H_poi_src_aug = get_hyper_deg(H_poi_src_aug)
        self.HG_poi_src_aug = transform_csr_matrix_to_tensor(
            Deg_H_poi_src_aug * H_poi_src_aug
        ).to(self.device)

        H_poi_tar_aug = H_poi_src_aug.T
        Deg_H_poi_tar_aug = get_hyper_deg(H_poi_tar_aug)
        self.HG_poi_tar_aug = transform_csr_matrix_to_tensor(
            Deg_H_poi_tar_aug * H_poi_tar_aug
        ).to(self.device)


class POIPartialDataset(Dataset):

    def __init__(self, full_dataset, user_indices):
        self.data = [full_dataset[i] for i in user_indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class POISessionDataset(Dataset):

    def __init__(
        self,
        data_filename,
        label_filename,
        pois_coos_filename,
        num_pois,
        padding_idx,
        args,
        device,
    ):

        self.sessions_dict = load_dict_from_pkl(data_filename)
        self.labels_dict = load_dict_from_pkl(label_filename)
        self.pois_coos_dict = load_dict_from_pkl(pois_coos_filename)
        self.users_trajs_dict = self.sessions_dict

        self.num_pois = num_pois
        self.num_sessions = len(self.sessions_dict)
        self.padding_idx = padding_idx
        self.distance_threshold = args.distance_threshold
        self.keep_rate = args.keep_rate
        self.device = device

        self.poi_geo_adj = gen_poi_geo_adj(
            num_pois, self.pois_coos_dict, self.distance_threshold
        )
        self.poi_geo_graph_matrix = normalized_adj(
            adj=self.poi_geo_adj, is_symmetric=False
        )
        self.poi_geo_graph = transform_csr_matrix_to_tensor(
            self.poi_geo_graph_matrix
        ).to(device)

        self.H_poi_src = gen_sparse_directed_H_poi(self.users_trajs_dict, num_pois)

        self.H_poi_src = csr_matrix_drop_edge(self.H_poi_src, args.keep_rate_poi)
        self.Deg_H_poi_src = get_hyper_deg(self.H_poi_src)
        self.HG_poi_src = self.Deg_H_poi_src * self.H_poi_src
        self.HG_poi_src = transform_csr_matrix_to_tensor(self.HG_poi_src).to(device)

        self.H_poi_tar = self.H_poi_src.T
        self.Deg_H_poi_tar = get_hyper_deg(self.H_poi_tar)
        self.HG_poi_tar = self.Deg_H_poi_tar * self.H_poi_tar
        self.HG_poi_tar = transform_csr_matrix_to_tensor(self.HG_poi_tar).to(device)

        self.H_poi_session = gen_sparse_H_pois_session(
            self.sessions_dict, num_pois, self.num_sessions
        )
        self.HG_col = gen_HG_from_sparse_H(self.H_poi_session)
        self.HG_col = transform_csr_matrix_to_tensor(self.HG_col).to(device)

        self.H_pu = self.H_poi_session

        self.H_up = self.H_pu.T
        self.Deg_H_up = get_hyper_deg(self.H_up)
        self.HG_up = self.Deg_H_up * self.H_up
        self.HG_up = transform_csr_matrix_to_tensor(self.HG_up).to(device)

    def __len__(self):

        return self.num_sessions

    def __getitem__(self, user_idx):
        user_seq = self.users_trajs_dict[user_idx]
        user_seq_len = len(user_seq)
        user_seq_mask = [1] * user_seq_len
        user_rev_seq = user_seq[::-1]
        label = self.labels_dict[user_idx]

        sample = {
            "user_idx": torch.tensor(user_idx).to(self.device),
            "user_seq": torch.tensor(user_seq).to(self.device),
            "user_rev_seq": torch.tensor(user_rev_seq).to(self.device),
            "user_seq_len": torch.tensor(user_seq_len).to(self.device),
            "user_seq_mask": torch.tensor(user_seq_mask).to(self.device),
            "label": torch.tensor(label).to(self.device),
        }

        return sample


def collate_fn_4sq(batch, padding_value=3835):

    batch_user_idx = []
    batch_user_seq = []
    batch_user_rev_seq = []
    batch_user_seq_len = []
    batch_user_seq_mask = []
    batch_label = []
    batch_user_time_seq = []
    for item in batch:
        batch_user_idx.append(item["user_idx"])
        batch_user_seq_len.append(item["user_seq_len"])
        batch_label.append(item["label"])
        batch_user_seq.append(item["user_seq"])
        batch_user_rev_seq.append(item["user_rev_seq"])
        batch_user_seq_mask.append(item["user_seq_mask"])
        if "user_time_seq" in item:
            batch_user_time_seq.append(item["user_time_seq"])

    pad_user_seq = pad_sequence(
        batch_user_seq, batch_first=True, padding_value=padding_value
    )
    pad_user_rev_seq = pad_sequence(
        batch_user_rev_seq, batch_first=True, padding_value=padding_value
    )
    pad_user_seq_mask = pad_sequence(
        batch_user_seq_mask, batch_first=True, padding_value=0
    )

    batch_user_idx = torch.stack(batch_user_idx)
    batch_user_seq_len = torch.stack(batch_user_seq_len)
    batch_label = torch.stack(batch_label)

    collate_sample = {
        "user_idx": batch_user_idx,
        "user_seq": pad_user_seq,
        "user_rev_seq": pad_user_rev_seq,
        "user_seq_len": batch_user_seq_len,
        "user_seq_mask": pad_user_seq_mask,
        "label": batch_label,
    }

    if len(batch_user_time_seq) > 0:
        pad_user_time_seq = pad_sequence(
            batch_user_time_seq, batch_first=True, padding_value=0
        )
        collate_sample["user_time_seq"] = pad_user_time_seq

    return collate_sample
