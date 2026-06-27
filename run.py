import argparse
import copy
import time
import os
import sys
import logging
import yaml
import datetime
import torch.optim as optim
import random

from dataset import *
from model import *
from metrics import batch_performance
from utils import *

try:
    from tqdm import tqdm
except ImportError:

    class tqdm:

        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        @staticmethod
        def write(message):
            print(message)


DATASET_DEFAULTS = {
    "NYC": {
        "ah1": False,
        "add_threshold": 0.1,
        "remove_threshold": 0,
        "ah2": False,
        "hg_keep_rate": 1,
        "h3_path_len": False,
        "add_time_data": False,
        "remove_cl": False,
        "seed": 2023,
        "distance_threshold": 2.5,
        "num_epochs": 100,
        "batch_size": 64,
        "emb_dim": 128,
        "lr": 0.001,
        "decay": 0.0005,
        "dropout": 0.3,
        "deviceID": 0,
        "lambda_cl": 0.1,
        "num_mv_layers": 3,
        "num_geo_layers": 3,
        "num_di_layers": 3,
        "temperature": 0.1,
        "keep_rate": 1,
        "keep_rate_poi": 1,
        "pdas_start_ratio": 0.3,
        "pdas_end_ratio": 0.1,
        "pdas_seed": None,
        "lr_scheduler_factor": 0.1,
        "save_dir": "logs",
    },
}


def _get_explicit_arg_dests(parser, argv):
    explicit_dests = set()
    for action in parser._actions:
        for option in action.option_strings:
            if option.startswith("--") and any(
                arg == option or arg.startswith(option + "=") for arg in argv
            ):
                explicit_dests.add(action.dest)
    return explicit_dests


def _apply_dataset_defaults(args, explicit_dests):
    dataset_defaults = DATASET_DEFAULTS.get(args.dataset)
    if not dataset_defaults:
        return args
    for key, value in dataset_defaults.items():
        if key not in explicit_dests:
            setattr(args, key, value)
    return args


torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="NYC", choices=("NYC",), help="NYC")


parser.add_argument("--ah1", action="store_true", help="adaptive interaction hg")
parser.add_argument("--add_threshold", type=float, default=0.1)
parser.add_argument("--remove_threshold", type=float, default=0)
parser.add_argument("--ah2", action="store_true", help="adaptive geo hg")

parser.add_argument("--hg_keep_rate", type=float, default=1)

parser.add_argument(
    "--h3_path_len", action="store_true", help="gen_path_length_weighted_H_poi()"
)

parser.add_argument("--add_time_data", action="store_true", help="add time_data")

parser.add_argument("--remove_cl", action="store_true", help="")

parser.add_argument("--seed", type=int, default=2023, help="Random seed")
parser.add_argument(
    "--distance_threshold",
    default=2.5,
    type=float,
    help="distance threshold 2.5 or 0.25",
)
parser.add_argument("--num_epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=200, help="input batch size")
parser.add_argument("--emb_dim", type=int, default=128, help="embedding size")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument(
    "--lambda_reg",
    "--decay",
    dest="decay",
    type=float,
    default=5e-4,
    help="lambda3 / Adam weight decay regularization coefficient",
)
parser.add_argument("--dropout", type=float, default=0.3, help="dropout")
parser.add_argument(
    "--cuda", "--deviceID", dest="deviceID", type=int, default=0, help="CUDA device id"
)
parser.add_argument(
    "--lambda_cl", type=float, default=0.1, help="lambda of contrastive loss"
)
parser.add_argument("--lambda_kl", type=float, default=1.0, help="lambda of KL loss")
parser.add_argument("--num_mv_layers", type=int, default=3)
parser.add_argument("--num_geo_layers", type=int, default=3)
parser.add_argument(
    "--num_di_layers",
    type=int,
    default=3,
    help="layer number of directed hypergraph convolutional network",
)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--keep_rate", type=float, default=1, help="ratio of edges to keep")
parser.add_argument(
    "--keep_rate_poi",
    type=float,
    default=1,
    help="ratio of poi-poi directed edges to keep",
)
parser.add_argument(
    "--noise_poi_ratio",
    type=float,
    default=0.0,
    help="random noisy POI insertion ratio for training trajectories",
)
parser.add_argument(
    "--noise_poi_seed",
    type=int,
    default=None,
    help="random seed for noisy POI insertion; default uses --seed",
)
parser.add_argument(
    "--pdas_start_ratio",
    type=float,
    default=0.3,
    help="initial PDAS perturbation ratio",
)
parser.add_argument(
    "--pdas_end_ratio", type=float, default=0.1, help="final PDAS perturbation ratio"
)
parser.add_argument(
    "--pdas_seed", type=int, default=None, help="random seed for PDAS perturbation"
)
parser.add_argument(
    "--lr-scheduler-factor",
    type=float,
    default=0.1,
    help="Learning rate scheduler factor",
)
parser.add_argument("--save_dir", type=str, default="logs")
args = parser.parse_args()
args = _apply_dataset_defaults(args, _get_explicit_arg_dests(parser, sys.argv[1:]))
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


global adaptive_bool
if args.ah1 or args.ah2:
    adaptive_bool = True
else:
    adaptive_bool = False


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


device = torch.device(
    "cuda:{}".format(args.deviceID) if torch.cuda.is_available() else "cpu"
)


current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
current_save_dir = os.path.join(args.save_dir, current_time)


os.mkdir(current_save_dir)


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=os.path.join(current_save_dir, f"log_training.txt"),
    filemode="w+",
)
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)


args_filename = args.dataset + "_args.yaml"
with open(os.path.join(current_save_dir, args_filename), "w") as f:
    yaml.dump(vars(args), f, sort_keys=False)


def mask(adj, epsilon=0, mask_value=-1e16):
    mask = (adj > epsilon).detach().float()
    update_adj = adj * mask + (1 - mask) * mask_value
    return update_adj


def main():

    logging.info("1. Parse Arguments")
    logging.info(args)
    logging.info("device: {}".format(device))
    NUM_USERS = 963
    NUM_POIS = 4628
    PADDING_IDX = NUM_POIS

    logging.info("2. Load Dataset")

    if args.add_time_data:
        train_dataset = POIDataset(
            data_filename="datasets/{}/train_poi_zero.txt".format(args.dataset),
            pois_coos_filename="datasets/{}/{}_pois_coos_poi_zero.pkl".format(
                args.dataset, args.dataset
            ),
            num_users=NUM_USERS,
            num_pois=NUM_POIS,
            padding_idx=PADDING_IDX,
            args=args,
            device=device,
            time_data_filename="datasets/{}/train_time_zero.txt".format(args.dataset),
            enable_noise_poi=True,
        )
        test_dataset = POIDataset(
            data_filename="datasets/{}/test_poi_zero.txt".format(args.dataset),
            pois_coos_filename="datasets/{}/{}_pois_coos_poi_zero.pkl".format(
                args.dataset, args.dataset
            ),
            num_users=NUM_USERS,
            num_pois=NUM_POIS,
            padding_idx=PADDING_IDX,
            args=args,
            device=device,
            time_data_filename="datasets/{}/test_time_zero.txt".format(args.dataset),
        )
    else:
        train_dataset = POIDataset(
            data_filename="datasets/{}/train_poi_zero.txt".format(args.dataset),
            pois_coos_filename="datasets/{}/{}_pois_coos_poi_zero.pkl".format(
                args.dataset, args.dataset
            ),
            num_users=NUM_USERS,
            num_pois=NUM_POIS,
            padding_idx=PADDING_IDX,
            args=args,
            device=device,
            enable_noise_poi=True,
        )
        test_dataset = POIDataset(
            data_filename="datasets/{}/test_poi_zero.txt".format(args.dataset),
            pois_coos_filename="datasets/{}/{}_pois_coos_poi_zero.pkl".format(
                args.dataset, args.dataset
            ),
            num_users=NUM_USERS,
            num_pois=NUM_POIS,
            padding_idx=PADDING_IDX,
            args=args,
            device=device,
        )

    noise_stats = getattr(train_dataset, "noise_poi_stats", None)
    if noise_stats and noise_stats["inserted_pois"] > 0:
        logging.info(
            "Random noisy POI injection: ratio={:.2f}, inserted_pois={}, changed_trajectories={}/{}".format(
                noise_stats["noise_ratio"],
                noise_stats["inserted_pois"],
                noise_stats["changed_trajectories"],
                noise_stats["trajectories"],
            )
        )

    logging.info("3. Construct DataLoader")

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_4sq(batch, padding_value=PADDING_IDX),
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_4sq(batch, padding_value=PADDING_IDX),
    )

    logging.info("4. Load Model")
    model = DCHL(NUM_USERS, NUM_POIS, args, device)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    criterion = nn.CrossEntropyLoss().to(device)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=args.lr_scheduler_factor
    )

    kl_criterion = torch.nn.KLDivLoss(reduction="batchmean")
    trans_poi_matrix = (
        transform_csr_matrix_to_tensor(train_dataset.H_poi_src).to_dense().to(device)
    )
    inter_poi_user_matrix = (
        transform_csr_matrix_to_tensor(train_dataset.H_pu).to_dense().to(device)
    )

    logging.info("5. Start Training")
    Ks_list = [1, 5, 10, 20]
    final_results = {
        "Rec1": 0.0,
        "Rec5": 0.0,
        "Rec10": 0.0,
        "Rec20": 0.0,
        "NDCG1": 0.0,
        "NDCG5": 0.0,
        "NDCG10": 0.0,
        "NDCG20": 0.0,
    }

    monitor_loss = float("inf")
    best_test_rec5 = 0.0

    early_stopping = EarlyStopping(patience=10, min_epochs=50, verbose=True)
    pdas_seed = args.seed if args.pdas_seed is None else args.pdas_seed

    train_loss_rec_ls = []
    train_loss_cl_pois_ls = []
    train_loss_cl_users_ls = []
    train_loss_ls = []
    test_loss_rec_ls = []
    test_loss_cl_pois_ls = []
    test_loss_cl_users_ls = []
    test_loss_ls = []
    for epoch in range(args.num_epochs):
        logging.info(
            "================= Epoch {}/{} =================".format(
                epoch, args.num_epochs
            )
        )
        start_time = time.time()
        model.train()

        pdas_ratio = pdas_ratio_for_epoch(
            epoch,
            args.num_epochs,
            start_ratio=args.pdas_start_ratio,
            end_ratio=args.pdas_end_ratio,
        )
        if args.hg_keep_rate < 1:
            train_dataset_epoch = copy.deepcopy(train_dataset)
            train_dataset_epoch.drop_edge_HG_pu(keep_rate=args.hg_keep_rate)
        else:
            train_dataset_epoch = train_dataset
        train_dataset_epoch.refresh_pdas_augmented_graphs(
            pdas_ratio,
            rng=np.random.default_rng(pdas_seed + epoch),
        )
        logging.info("PDAS perturbation ratio: {:.4f}".format(pdas_ratio))

        train_loss = 0.0

        train_recall_array = np.zeros(shape=(len(train_dataloader), len(Ks_list)))
        train_ndcg_array = np.zeros(shape=(len(train_dataloader), len(Ks_list)))
        train_batches = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc="Epoch {}/{} Train".format(epoch + 1, args.num_epochs),
            leave=False,
        )
        for idx, batch in train_batches:
            logging.info("Train. Batch {}/{}".format(idx, len(train_dataloader)))
            optimizer.zero_grad()

            predictions, loss_cl_users, loss_cl_pois = model(train_dataset_epoch, batch)

            loss_rec = criterion(predictions, batch["label"].to(device))

            if epoch != 0 and adaptive_bool:

                user_emds = model.get_user_embedding()
                poi_embs = model.get_poi_embedding()
                geo_poi_embs = model.get_geo_poi_embedding()
                seq_poi_embs = model.get_seq_poi_embedding()
                seq_user_embs = model.get_seq_user_embedding()

                geo_similarity_matrix = calculate_similarity(geo_poi_embs)
                seq_similarity_matrix = calculate_similarity(seq_poi_embs)

                kl_loss1 = kl_criterion(
                    torch.log(
                        torch.softmax(mask(geo_similarity_matrix), dim=-1) + 1e-9
                    ),
                    torch.softmax(mask(trans_poi_matrix), dim=-1),
                )
                kl_loss2 = kl_criterion(
                    torch.log(
                        torch.softmax(mask(seq_similarity_matrix), dim=-1) + 1e-9
                    ),
                    torch.softmax(mask(trans_poi_matrix), dim=-1),
                )
                kl_loss = (kl_loss1 + kl_loss2) / 2

                loss = (
                    loss_rec
                    + args.lambda_cl * (loss_cl_pois + loss_cl_users)
                    + args.lambda_kl * kl_loss
                )
                if args.remove_cl:
                    loss = loss_rec + args.lambda_kl * kl_loss

                logging.info(
                    "Train. loss_rec: {:.4f}; loss_cl_pois: {:.4f}; loss_cl_users: {:.4f}; kl_loss: {:.4f};"
                    " loss: {:.4f}".format(
                        loss_rec.item(), loss_cl_pois, loss_cl_users, kl_loss, loss
                    )
                )
            else:
                loss = loss_rec + args.lambda_cl * (loss_cl_pois + loss_cl_users)
                logging.info(
                    "Train. loss_rec: {:.4f}; loss_cl_pois: {:.4f}; loss_cl_users: {:.4f}; "
                    "loss: {:.4f}".format(
                        loss_rec.item(), loss_cl_pois, loss_cl_users, loss
                    )
                )
                if args.remove_cl:
                    loss = loss_rec

            train_loss_rec_ls.append(loss_rec.item())
            train_loss_cl_pois_ls.append(loss_cl_pois.item())
            train_loss_cl_users_ls.append(loss_cl_users.item())
            train_loss_ls.append(loss.item())

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            for k in Ks_list:
                recall, ndcg = batch_performance(
                    predictions.detach().cpu(), batch["label"].detach().cpu(), k
                )
                col_idx = Ks_list.index(k)
                train_recall_array[idx, col_idx] = recall
                train_ndcg_array[idx, col_idx] = ndcg
        logging.info(
            "Training finishes at this epoch. It takes {} min".format(
                (time.time() - start_time) / 60
            )
        )
        logging.info("Training loss: {:.4f}".format(train_loss / len(train_dataloader)))
        logging.info("Training Epoch {}/{} results:".format(epoch, args.num_epochs))
        for k in Ks_list:
            col_idx = Ks_list.index(k)
            logging.info(
                "Recall@{}: {:.4f}".format(k, np.mean(train_recall_array[:, col_idx]))
            )
            logging.info(
                "NDCG@{}: {:.4f}".format(k, np.mean(train_ndcg_array[:, col_idx]))
            )
        logging.info("\n")

        logging.info("Testing")
        test_loss = 0.0
        test_recall_array = np.zeros(shape=(len(test_dataloader), len(Ks_list)))
        test_ndcg_array = np.zeros(shape=(len(test_dataloader), len(Ks_list)))

        if epoch != 0 and adaptive_bool:

            if args.ah1:
                train_dataset.update_poi_session_hypergraph(
                    seq_similarity_matrix,
                    add_threshold=args.add_threshold,
                    remove_threshold=args.remove_threshold,
                )
                test_dataset.update_poi_session_hypergraph(
                    seq_similarity_matrix,
                    add_threshold=args.add_threshold,
                    remove_threshold=args.remove_threshold,
                )

            if args.ah2:
                train_dataset.update_poi_geographical_hypergraph(geo_similarity_matrix)
                test_dataset.update_poi_geographical_hypergraph(geo_similarity_matrix)

        model.eval()
        with torch.no_grad():
            test_batches = tqdm(
                enumerate(test_dataloader),
                total=len(test_dataloader),
                desc="Epoch {}/{} Test".format(epoch + 1, args.num_epochs),
                leave=False,
            )
            for idx, batch in test_batches:

                logging.info("Test. Batch {}/{}".format(idx, len(test_dataloader)))

                predictions, loss_cl_users, loss_cl_pois = model(test_dataset, batch)

                loss_rec = criterion(predictions, batch["label"].to(device))
                loss = loss_rec + args.lambda_cl * (loss_cl_pois + loss_cl_users)
                logging.info(
                    "Test. loss_rec: {:.4f}; loss_cl_pois: {:.4f}; loss_cl_users: {:.4f}; "
                    "loss: {:.4f}".format(
                        loss_rec.item(), loss_cl_pois, loss_cl_users, loss
                    )
                )

                test_loss_rec_ls.append(loss_rec.item())
                test_loss_cl_pois_ls.append(loss_cl_pois.item())
                test_loss_cl_users_ls.append(loss_cl_users.item())
                test_loss_ls.append(loss.item())

                test_loss += loss.item()

                for k in Ks_list:
                    recall, ndcg = batch_performance(
                        predictions.detach().cpu(), batch["label"].detach().cpu(), k
                    )
                    col_idx = Ks_list.index(k)
                    test_recall_array[idx, col_idx] = recall
                    test_ndcg_array[idx, col_idx] = ndcg
        logging.info("Testing finishes")
        logging.info("Testing loss: {}".format(test_loss / len(test_dataloader)))
        logging.info("Testing results:")
        test_metrics = {}
        for k in Ks_list:
            col_idx = Ks_list.index(k)
            recall = np.mean(test_recall_array[:, col_idx])
            ndcg = np.mean(test_ndcg_array[:, col_idx])
            test_metrics["R{}".format(k)] = recall
            test_metrics["N{}".format(k)] = ndcg
            logging.info("Recall@{}: {:.4f}".format(k, recall))
            logging.info("NDCG@{}: {:.4f}".format(k, ndcg))
        tqdm.write(
            "Epoch {}/{} Test | R5: {:.4f} R10: {:.4f} N5: {:.4f} N10: {:.4f}".format(
                epoch + 1,
                args.num_epochs,
                test_metrics["R5"],
                test_metrics["R10"],
                test_metrics["N5"],
                test_metrics["N10"],
            )
        )

        monitor_loss = min(monitor_loss, test_loss)

        lr_scheduler.step(monitor_loss)

        test_recall5 = np.mean(test_recall_array[:, 1])

        early_stopping(test_recall5, epoch)

        if early_stopping.early_stop:
            logging.info("Early stopping at epoch {}".format(epoch))
            break

        if test_recall5 > best_test_rec5:
            best_test_rec5 = test_recall5
            logging.info("Update test results and save model at epoch{}".format(epoch))

            saved_model_path = os.path.join(
                current_save_dir, "{}.pt".format(args.dataset)
            )
            torch.save(model.state_dict(), saved_model_path)

        for k in Ks_list:
            if k == 1:
                final_results["Rec1"] = max(
                    final_results["Rec1"], np.mean(test_recall_array[:, 0])
                )
                final_results["NDCG1"] = max(
                    final_results["NDCG1"], np.mean(test_ndcg_array[:, 0])
                )

            elif k == 5:
                final_results["Rec5"] = max(
                    final_results["Rec5"], np.mean(test_recall_array[:, 1])
                )
                final_results["NDCG5"] = max(
                    final_results["NDCG5"], np.mean(test_ndcg_array[:, 1])
                )

            elif k == 10:
                final_results["Rec10"] = max(
                    final_results["Rec10"], np.mean(test_recall_array[:, 2])
                )
                final_results["NDCG10"] = max(
                    final_results["NDCG10"], np.mean(test_ndcg_array[:, 2])
                )

            elif k == 20:
                final_results["Rec20"] = max(
                    final_results["Rec20"], np.mean(test_recall_array[:, 3])
                )
                final_results["NDCG20"] = max(
                    final_results["NDCG20"], np.mean(test_ndcg_array[:, 3])
                )
        logging.info("==================================\n\n")

    logging.info("6. Final Results")
    formatted_dict = {key: f"{value:.4f}" for key, value in final_results.items()}
    logging.info(formatted_dict)
    logging.info("\n")
    tqdm.write(
        "Final Results | R5: {:.4f} R10: {:.4f} N5: {:.4f} N10: {:.4f}".format(
            final_results["Rec5"],
            final_results["Rec10"],
            final_results["NDCG5"],
            final_results["NDCG10"],
        )
    )

    logging.info("train_loss_rec_ls: %s", train_loss_rec_ls)
    logging.info("train_loss_cl_pois_ls: %s", train_loss_cl_pois_ls)
    logging.info("train_loss_cl_users_ls: %s", train_loss_cl_users_ls)
    logging.info("train_loss_ls: %s", train_loss_ls)
    logging.info("test_loss_rec_ls: %s", test_loss_rec_ls)
    logging.info("test_loss_cl_pois_ls: %s", test_loss_cl_pois_ls)
    logging.info("test_loss_cl_users_ls: %s", test_loss_cl_users_ls)
    logging.info("test_loss_ls: %s", test_loss_ls)


if __name__ == "__main__":
    main()
