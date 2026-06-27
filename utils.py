import pickle
from collections import defaultdict

import numpy as np
from math import radians, cos, sin, asin, sqrt
import scipy.sparse as sp
import torch


def get_unique_seq(sessions_list):
    seq_list = []
    for session in sessions_list:
        for poi in session:
            if poi in seq_list:
                continue
            else:
                seq_list.append(poi)

    return seq_list


def get_unique_seqs_for_sessions(sessions_dict):
    seqs_dict = {}
    seqs_lens_dict = {}
    for key, value in sessions_dict.items():
        seqs_dict[key] = get_unique_seq(value)
        seqs_lens_dict[key] = len(get_unique_seq(value))

    return seqs_dict, seqs_lens_dict


def get_seqs_for_sessions(sessions_dict, padding_idx, max_seq_len):
    seqs_dict = {}
    seqs_lens_dict = {}
    reverse_seqs_dict = {}
    for key, sessions in sessions_dict.items():
        temp = []
        for session in sessions:
            temp.extend(session)
        if len(temp) >= max_seq_len:
            temp = temp[-max_seq_len:]
            temp_rev = temp[::-1]
            seqs_dict[key] = temp
            reverse_seqs_dict[key] = temp_rev
            seqs_lens_dict[key] = max_seq_len
        else:
            temp_new = temp + [padding_idx] * (max_seq_len - len(temp))
            temp_rev = temp[::-1] + [padding_idx] * (max_seq_len - len(temp))
            seqs_dict[key] = temp_new
            reverse_seqs_dict[key] = temp_rev
            seqs_lens_dict[key] = len(temp)

    return seqs_dict, reverse_seqs_dict, seqs_lens_dict


def save_list_with_pkl(filename, list_obj):
    with open(filename, "wb") as f:
        pickle.dump(list_obj, f)


def load_list_with_pkl(filename):
    with open(filename, "rb") as f:
        list_obj = pickle.load(f)

    return list_obj


def save_dict_to_pkl(pkl_filename, dict_pbj):
    with open(pkl_filename, "wb") as f:
        pickle.dump(dict_pbj, f)


def load_dict_from_pkl(pkl_filename):
    with open(pkl_filename, "rb") as f:
        dict_obj = pickle.load(f)

    return dict_obj


def get_num_sessions(sessions_dict):
    num_sessions = 0
    for value in sessions_dict.values():
        num_sessions += len(value)

    return num_sessions


def get_user_complete_traj(sessions_dict):
    users_trajs_dict = {}
    users_trajs_lens_dict = {}
    for userID, sessions in sessions_dict.items():
        traj = []
        for session in sessions:
            traj.extend(session)
        users_trajs_dict[userID] = traj
        users_trajs_lens_dict[userID] = len(traj)

    return users_trajs_dict, users_trajs_lens_dict


def get_user_complete_traj_2(sessions_dict, train_sessions_dict, train_labels_dict):

    users_trajs_dict = {}
    users_trajs_lens_dict = {}
    for userID, sessions in sessions_dict.items():
        traj = []
        for train_session in train_sessions_dict[userID]:
            traj.extend(train_session)
        traj.append(train_labels_dict[userID])
        for session in sessions:
            traj.extend(session)
        users_trajs_dict[userID] = traj
        users_trajs_lens_dict[userID] = len(traj)

    return users_trajs_dict, users_trajs_lens_dict


def get_user_reverse_traj(users_trajs_dict):
    users_rev_trajs_dict = {}
    for userID, traj in users_trajs_dict.items():
        rev_traj = traj[::-1]
        users_rev_trajs_dict[userID] = rev_traj

    return users_rev_trajs_dict


def gen_poi_geo_adj(num_pois, pois_coos_dict, distance_threshold):
    poi_geo_adj = np.zeros(shape=(num_pois, num_pois))

    for poi1 in range(num_pois):
        lat1, lon1 = pois_coos_dict[poi1]
        for poi2 in range(poi1, num_pois):
            lat2, lon2 = pois_coos_dict[poi2]
            hav_dist = haversine_distance(lon1, lat1, lon2, lat2)
            if hav_dist <= distance_threshold:
                poi_geo_adj[poi1, poi2] = 1
                poi_geo_adj[poi2, poi1] = 1

    poi_geo_adj = sp.csr_matrix(poi_geo_adj)

    return poi_geo_adj


def process_users_seqs(users_seqs_dict, padding_idx, max_seq_len):
    processed_seqs_dict = {}
    reverse_seqs_dict = {}
    for key, seq in users_seqs_dict.items():
        if len(seq) >= max_seq_len:
            temp_seq = seq[-max_seq_len:]
            temp_rev_seq = temp_seq[::-1]
        else:
            temp_seq = seq + [padding_idx] * (max_seq_len - len(seq))
            temp_rev_seq = seq[::-1] + [padding_idx] * (max_seq_len - len(seq))
        processed_seqs_dict[key] = temp_seq
        reverse_seqs_dict[key] = temp_rev_seq

    return processed_seqs_dict, reverse_seqs_dict


def reverse_users_seqs(processed_users_seqs_dict, padding_idx, max_seq_len):
    reversed_users_seqs_dict = {}
    for key, seq in processed_users_seqs_dict.items():
        for idx in range(len(seq)):
            if seq[idx] == padding_idx:
                actual_seq = seq[:idx]
                reversed_users_seqs_dict[key] = actual_seq[::-1] + [padding_idx] * (
                    max_seq_len - idx
                )
                break

    return reversed_users_seqs_dict


def gen_users_seqs_masks(users_seqs_dict, padding_idx):
    users_seqs_masks_dict = {}
    for key, seq in users_seqs_dict.items():
        temp_seq = []
        for poi in seq:
            if poi != padding_idx:
                temp_seq.append(1)
            else:
                temp_seq.append(0)
        users_seqs_masks_dict[key] = temp_seq

    return users_seqs_masks_dict


def haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371

    return c * r


def euclidean_distance(lon1, lat1, lon2, lat2):

    return np.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2)


def gen_geo_seqs_adjs_dict(
    users_seqs_dict,
    pois_coos_dict,
    max_seq_len,
    padding_idx,
    eta=1,
    distance_threshold=2.5,
    distance_type="haversine",
):
    geo_adjs_dict = {}
    for key, seq in users_seqs_dict.items():
        geo_adj = np.zeros(shape=(max_seq_len, max_seq_len))
        actual_seq = []
        for item in seq:
            if item != padding_idx:
                actual_seq.append(item)
        actual_seq_len = len(actual_seq)
        for i in range(actual_seq_len):
            for j in range(i + 1, actual_seq_len):
                l1 = actual_seq[i]
                l2 = actual_seq[j]
                lat1, lon1 = pois_coos_dict[l1]
                lat2, lon2 = pois_coos_dict[l2]
                if distance_type == "haversine":
                    dist = haversine_distance(lon1, lat1, lon2, lat2)
                elif distance_type == "euclidean":
                    dist = euclidean_distance(lon1, lat1, lon2, lat2)
                if 0 < dist <= distance_threshold:
                    geo_influence = np.exp(-eta * (dist**2))
                    geo_adj[i, j] = geo_influence
                    geo_adj[j, i] = geo_influence
        geo_adjs_dict[key] = geo_adj

    return geo_adjs_dict


def create_user_poi_adj(users_seqs_dict, num_users, num_pois):
    R = sp.dok_matrix((num_users, num_pois), dtype=np.float)
    for userID, seq in users_seqs_dict.items():
        for itemID in seq:
            itemID = itemID - num_users
            R[userID, itemID] = 1

    return R, R.T


def gen_sparse_interaction_matrix(users_seqs_dict, num_users, num_pois):
    R, R_T = create_user_poi_adj(users_seqs_dict, num_users, num_pois)
    A = sp.dok_matrix((num_users + num_pois, num_users + num_pois), dtype=float)
    A[:num_users, num_users:] = R
    A[num_users:, :num_users] = R_T
    A_sparse = A.tocsr()

    return A_sparse


def normalized_adj(adj, is_symmetric=True):
    if is_symmetric:
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum + 1e-8, -1 / 2).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv * adj * d_mat_inv
    else:
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum + 1e-8, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv * adj

    return norm_adj


def normalized_adj_tensor(adj_tensor):

    degree_tensor = torch.diag(torch.sum(adj_tensor, dim=1))

    inverse_degree_tensor = torch.inverse(degree_tensor)

    norm_adj = torch.matmul(inverse_degree_tensor, adj_tensor)

    sparse_norm_adj = torch.sparse.FloatTensor(norm_adj)

    return sparse_norm_adj


def gen_local_graph(adj):
    G = normalized_adj(adj + sp.eye(adj.shape[0]))

    return G


def gen_sparse_H(sessions_dict, num_pois, num_sessions, start_poiID):
    H = np.zeros(shape=(num_pois, num_sessions))
    sess_idx = 0
    for key, sessions in sessions_dict.items():
        for session in sessions:
            for poiID in session:
                new_poiID = poiID - start_poiID
                H[new_poiID, sess_idx] = 1
            sess_idx += 1
    assert sess_idx == num_sessions
    H = sp.csr_matrix(H)

    return H


def gen_sparse_H_pois_session(sessions_dict, num_pois, num_sessions):
    H = np.zeros(shape=(num_pois, num_sessions))
    for sess_idx, session in sessions_dict.items():
        for poi in session:
            H[poi, sess_idx] = 1
    H = sp.csr_matrix(H)

    return H


def gen_sparse_H_user(sessions_dict, num_pois, num_users):
    H = np.zeros(shape=(num_pois, num_users))

    for userID, sessions in sessions_dict.items():
        seq = []
        for session in sessions:
            seq.extend(session)
        for poi in seq:
            H[poi, userID] = 1

    H = sp.csr_matrix(H)

    return H


def gen_sparse_H_user_traj(traj_dict, num_pois, num_users):
    H = np.zeros(shape=(num_pois, num_users))

    for userID, traj in traj_dict.items():
        seq = []
        seq.extend(traj)
        for poi in seq:
            H[poi, userID] = 1

    H = sp.csr_matrix(H)

    return H


def gen_poi_cooccurrence_adj(H_pu):
    poi_adj = H_pu.dot(H_pu.T).astype(np.float64).tocsr()
    poi_adj.setdiag(0)
    poi_adj.eliminate_zeros()
    return poi_adj


def gen_sparse_directed_H_poi(users_trajs_dict, num_pois):
    H = np.zeros(shape=(num_pois, num_pois))
    for userID, traj in users_trajs_dict.items():
        for src_idx in range(len(traj) - 1):
            for tar_idx in range(src_idx + 1, len(traj)):
                src_poi = traj[src_idx]
                tar_poi = traj[tar_idx]
                H[src_poi, tar_poi] = 1

    H = sp.csr_matrix(H)

    return H


def gen_time_decay_H_poi(users_trajs_dict, num_pois, users_time_dict):
    H = np.zeros(shape=(num_pois, num_pois))
    for userID, traj in users_trajs_dict.items():
        times = users_time_dict[userID]
        for src_idx in range(len(traj) - 1):
            for tar_idx in range(src_idx + 1, len(traj)):
                src_poi = traj[src_idx]
                tar_poi = traj[tar_idx]
                time_diff = abs(times[tar_idx] - times[src_idx])
                if time_diff == 0:
                    time_diff = 1
                H[src_poi, tar_poi] += 1 / time_diff

    H = sp.csr_matrix(H)
    return H


def gen_inverse_freq_weighted_H_poi(users_trajs_dict, num_pois):
    H = np.zeros(shape=(num_pois, num_pois))
    freq_dict = defaultdict(int)

    for userID, traj in users_trajs_dict.items():
        for src_idx in range(len(traj) - 1):
            for tar_idx in range(src_idx + 1, len(traj)):
                src_poi = traj[src_idx]
                tar_poi = traj[tar_idx]
                freq_dict[(src_poi, tar_poi)] += 1

    for (src_poi, tar_poi), freq in freq_dict.items():
        H[src_poi, tar_poi] = 1 / freq

    H = sp.csr_matrix(H)
    return H


def gen_path_length_weighted_H_poi(users_trajs_dict, num_pois):
    H = np.zeros(shape=(num_pois, num_pois))
    for userID, traj in users_trajs_dict.items():
        for src_idx in range(len(traj) - 1):
            for tar_idx in range(src_idx + 1, len(traj)):
                src_poi = traj[src_idx]
                tar_poi = traj[tar_idx]
                path_length = tar_idx - src_idx
                H[src_poi, tar_poi] += 1 / path_length

    H = sp.csr_matrix(H)
    return H


def gen_HG_from_sparse_H(H, conv="sym"):
    n_edge = H.shape[1]
    W = sp.eye(n_edge)

    HW = H.dot(W)
    DV = sp.csr_matrix(HW.sum(axis=1)).astype(float)
    DE = sp.csr_matrix(H.sum(axis=0)).astype(float)
    invDE1 = DE.power(-1)
    invDE1_ = sp.diags(invDE1.toarray()[0])
    HT = H.T

    if conv == "sym":
        invDV2 = DV.power(n=-1 / 2)
        invDV2_ = sp.diags(invDV2.toarray()[:, 0])
        HG = invDV2_ * H * W * invDE1_ * HT * invDV2_
    elif conv == "asym":
        invDV1 = DV.power(-1)
        invDV1_ = sp.diags(invDV1.toarray()[:, 0])
        HG = invDV1_ * H * W * invDE1_ * HT

    return HG


def get_hyper_deg(incidence_matrix):

    rowsum = np.array(incidence_matrix.sum(1))
    with np.errstate(divide="ignore"):
        d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)

    return d_mat_inv


def transform_csr_matrix_to_tensor(csr_matrix):
    coo = csr_matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    sp_tensor = torch.sparse_coo_tensor(i, v, torch.Size(shape))

    return sp_tensor


def get_poi_session_freq(num_pois, num_sessions, sessions_dict):
    poi_sess_freq_matrix = np.zeros(shape=(num_pois, num_sessions))

    sess_idx = 0
    for userID, sessions in sessions_dict.items():
        for session in sessions:
            for poiID in session:
                poi_sess_freq_matrix[poiID, sess_idx] += 1
            sess_idx += 1

    poi_sess_freq_matrix = sp.csr_matrix(poi_sess_freq_matrix)

    return poi_sess_freq_matrix


def get_all_sessions(sessions_dict):
    all_sessions = []

    for userID, sessions in sessions_dict.items():
        for session in sessions:
            all_sessions.append(torch.tensor(session))

    return all_sessions


def get_all_users_seqs(users_trajs_dict):
    all_seqs = []
    for userID, traj in users_trajs_dict.items():
        all_seqs.append(torch.tensor(traj))

    return all_seqs


def sparse_adj_tensor_drop_edge(sp_adj, keep_rate):
    if keep_rate == 1.0:
        return sp_adj

    vals = sp_adj._values()
    idxs = sp_adj._indices()
    edgeNum = vals.size()
    mask = ((torch.rand(edgeNum) + keep_rate).floor()).type(torch.bool)
    newVals = vals[mask] / keep_rate
    newIdxs = idxs[:, mask]

    return torch.sparse.FloatTensor(newIdxs, newVals, sp_adj.shape)


def csr_matrix_drop_edge(csr_adj_matrix, keep_rate=1):
    if keep_rate == 1.0:
        return csr_adj_matrix

    coo = csr_adj_matrix.tocoo()
    row = coo.row
    col = coo.col
    edgeNum = row.shape[0]

    mask = np.floor(np.random.rand(edgeNum) + keep_rate).astype(np.bool_)

    new_row = row[mask]
    new_col = col[mask]
    new_edgeNum = new_row.shape[0]
    new_values = np.ones(new_edgeNum, dtype=float)

    drop_adj_matrix = sp.csr_matrix((new_values, (new_row, new_col)), shape=coo.shape)

    return drop_adj_matrix


def pdas_ratio_for_epoch(epoch, num_epochs, start_ratio=0.3, end_ratio=0.1):
    if num_epochs <= 1:
        return float(start_ratio)
    progress = min(max(epoch, 0), num_epochs - 1) / (num_epochs - 1)
    ratio = start_ratio + (end_ratio - start_ratio) * progress
    return round(float(ratio), 10)


def pdas_perturb_incidence_matrix(incidence_matrix, perturb_ratio, rng=None):
    if not sp.issparse(incidence_matrix):
        incidence_matrix = sp.csr_matrix(incidence_matrix)
    if perturb_ratio <= 0:
        return incidence_matrix.copy().tocsr()

    ratio = min(float(perturb_ratio), 1.0)
    rng = np.random.default_rng() if rng is None else rng
    csc = incidence_matrix.tocsc(copy=True).astype(np.float64)
    num_rows, num_cols = csc.shape
    if num_rows == 0 or num_cols == 0:
        return csc.tocsr()

    selected_col_count = min(num_cols, int(np.ceil(ratio * num_cols)))
    if selected_col_count <= 0:
        return csc.tocsr()

    selected_cols = set(
        rng.choice(num_cols, size=selected_col_count, replace=False).tolist()
    )
    rows = []
    cols = []
    data = []
    all_rows = np.arange(num_rows)

    for col in range(num_cols):
        start = csc.indptr[col]
        end = csc.indptr[col + 1]
        nz_rows = csc.indices[start:end].copy()
        nz_data = csc.data[start:end].copy()

        if col in selected_cols and nz_rows.size > 0:
            zero_mask = np.ones(num_rows, dtype=bool)
            zero_mask[nz_rows] = False
            zero_rows = all_rows[zero_mask]
            swap_count = min(int(np.ceil(ratio * nz_rows.size)), zero_rows.size)
            if swap_count > 0:
                selected_nz_positions = rng.choice(
                    nz_rows.size, size=swap_count, replace=False
                )
                selected_zero_rows = rng.choice(
                    zero_rows, size=swap_count, replace=False
                )
                keep_mask = np.ones(nz_rows.size, dtype=bool)
                keep_mask[selected_nz_positions] = False
                moved_data = nz_data[selected_nz_positions].copy()
                nz_rows = np.concatenate([nz_rows[keep_mask], selected_zero_rows])
                nz_data = np.concatenate([nz_data[keep_mask], moved_data])

        rows.extend(nz_rows.tolist())
        cols.extend([col] * nz_rows.size)
        data.extend(nz_data.tolist())

    return sp.csr_matrix((data, (rows, cols)), shape=csc.shape)


def calculate_similarity(poi_embs):

    norm = poi_embs.norm(p=2, dim=1, keepdim=True)

    normalized_embs = poi_embs / norm

    similarity_matrix = torch.mm(normalized_embs, normalized_embs.T)

    return similarity_matrix


def calculate_similarity_pu(poi_embs, user_embs):

    eps = 1e-8
    poi_norm = poi_embs.norm(p=2, dim=1, keepdim=True).clamp(min=eps)
    user_norm = user_embs.norm(p=2, dim=1, keepdim=True).clamp(min=eps)

    normalized_poi_embs = poi_embs / poi_norm
    normalized_user_embs = user_embs / user_norm

    similarity_matrix = torch.mm(normalized_poi_embs, normalized_user_embs.T)

    return similarity_matrix


def update_hypergraph(similarity_matrix, threshold=0.5):

    new_edges = (similarity_matrix > threshold).float()

    new_edges.fill_diagonal_(0)
    return new_edges


def graph_regularization_loss(
    hypergraph,
    user_embeddings,
    poi_embeddings,
    lambda_sparse=1e-4,
    lambda_smooth=1e-3,
    lambda_conn=1e-4,
):

    def graph_sparsity_regularization(hypergraph):
        loss = torch.sum(torch.abs(hypergraph))
        return loss

    def graph_smoothness_regularization(hypergraph, node_embeddings):
        degree_matrix = torch.diag(torch.sum(hypergraph, dim=1))
        laplacian = degree_matrix - hypergraph
        smoothness_loss = torch.trace(
            torch.matmul(torch.matmul(node_embeddings.T, laplacian), node_embeddings)
        )
        return smoothness_loss

    def graph_connectivity_regularization(hypergraph):
        node_degrees = torch.sum(hypergraph, dim=1)
        isolated_nodes = torch.sum((node_degrees == 0).float())
        return isolated_nodes

    sparse_loss = graph_sparsity_regularization(hypergraph)
    smooth_loss_users = graph_smoothness_regularization(hypergraph, user_embeddings)
    smooth_loss_pois = graph_smoothness_regularization(hypergraph.T, poi_embeddings)
    connectivity_loss = graph_connectivity_regularization(hypergraph)

    total_loss = (
        lambda_sparse * sparse_loss
        + lambda_smooth * (smooth_loss_users + smooth_loss_pois)
        + lambda_conn * connectivity_loss
    )

    return total_loss


def spectral_regularization(features):

    frobenius_norm = torch.norm(features, p="fro")
    return frobenius_norm


class EarlyStopping:

    def __init__(self, patience=10, min_epochs=50, verbose=False, delta=0):
        self.patience = patience
        self.min_epochs = min_epochs
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.best_rec5 = -float("inf")

    def __call__(self, rec5, epoch):

        if epoch < self.min_epochs:
            return

        score = rec5

        if self.best_score is None:
            self.best_score = score
            self.best_rec5 = rec5
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_rec5 = rec5
            self.counter = 0
