import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import *


class MultiViewHyperConvLayer(nn.Module):

    def __init__(self, emb_dim, device):
        super(MultiViewHyperConvLayer, self).__init__()

        self.dropout = nn.Dropout(0.3)
        self.emb_dim = emb_dim
        self.device = device

    def forward(self, pois_embs, pad_all_train_sessions, HG_up, HG_pu):

        msg_poi_agg = torch.sparse.mm(HG_up, pois_embs)

        propag_pois_embs = torch.sparse.mm(HG_pu, msg_poi_agg)

        return propag_pois_embs


class DirectedHyperConvLayer(nn.Module):

    def __init__(self):
        super(DirectedHyperConvLayer, self).__init__()
        self.fc = nn.Linear(128, 128)
        self.gate = nn.Sigmoid()

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        msg_tar = torch.sparse.mm(HG_poi_tar, pois_embs)
        msg_src = torch.sparse.mm(HG_poi_src, msg_tar)

        return msg_src


class MultiViewHyperConvNetwork(nn.Module):

    def __init__(self, num_layers, emb_dim, dropout, device):
        super(MultiViewHyperConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.device = device
        self.mv_hconv_layer = MultiViewHyperConvLayer(emb_dim, device)
        self.dropout = dropout

    def forward(self, pois_embs, pad_all_train_sessions, HG_up, HG_pu):
        final_pois_embs = [pois_embs]
        for layer_idx in range(self.num_layers):
            pois_embs = self.mv_hconv_layer(
                pois_embs, pad_all_train_sessions, HG_up, HG_pu
            )

            pois_embs = pois_embs + final_pois_embs[-1]
            pois_embs = F.dropout(pois_embs, self.dropout, training=self.training)
            final_pois_embs.append(pois_embs)
        final_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)

        return final_pois_embs


class DirectedHyperConvNetwork(nn.Module):
    def __init__(self, num_layers, device, dropout=0.3):
        super(DirectedHyperConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.device = device
        self.dropout = dropout
        self.di_hconv_layer = DirectedHyperConvLayer()

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        final_pois_embs = [pois_embs]
        for layer_idx in range(self.num_layers):
            pois_embs = self.di_hconv_layer(pois_embs, HG_poi_src, HG_poi_tar)

            pois_embs = pois_embs + final_pois_embs[-1]
            pois_embs = F.dropout(pois_embs, self.dropout, training=self.training)
            final_pois_embs.append(pois_embs)
        final_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)

        return final_pois_embs


class GeoConvNetwork(nn.Module):
    def __init__(self, num_layers, dropout):
        super(GeoConvNetwork, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, pois_embs, geo_graph):
        final_pois_embs = [pois_embs]
        for _ in range(self.num_layers):

            pois_embs = torch.sparse.mm(geo_graph, pois_embs)

            final_pois_embs.append(pois_embs)
        output_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)

        return output_pois_embs


class SequenceModel(nn.Module):
    def __init__(
        self, input_dim=128, hidden_dim=128, num_heads=1, num_layers=1, time_emb_dim=128
    ):
        super(SequenceModel, self).__init__()

        self.time_embedding = SinusoidalTimeEncoding(time_emb_dim=time_emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim + time_emb_dim, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.fc = nn.Linear(input_dim + time_emb_dim, hidden_dim)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, seq_data, seq_time_data):

        seq_time_emb = self.time_embedding(seq_time_data)

        combined_data = torch.cat([seq_data, seq_time_emb], dim=-1)

        transformer_out = self.transformer_encoder(combined_data)

        sequence_embedding = transformer_out.mean(dim=1)

        sequence_embedding = self.fc(sequence_embedding)

        output = self.dropout(sequence_embedding)

        return output


class TemporalAttentionModel(nn.Module):
    def __init__(self, user_emb_dim=128):
        super(TemporalAttentionModel, self).__init__()

        self.fc = nn.Linear(user_emb_dim, user_emb_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, graph_user_embs, user_seq_embs):

        attn_weights = torch.sum(graph_user_embs * user_seq_embs, dim=-1)

        attn_weights = torch.sigmoid(attn_weights)

        user_embs = (
            attn_weights.unsqueeze(1) * graph_user_embs
            + (1 - attn_weights).unsqueeze(1) * user_seq_embs
        )

        user_embs = self.fc(user_embs)
        user_embs = self.dropout(user_embs)

        return user_embs


class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, time_emb_dim):
        super(SinusoidalTimeEncoding, self).__init__()
        self.time_emb_dim = time_emb_dim

    def forward(self, seq_time_data):
        batch_size, seq_len = seq_time_data.shape
        pe = torch.zeros(seq_len, self.time_emb_dim, device=seq_time_data.device)
        position = torch.arange(
            0, seq_len, dtype=torch.float, device=seq_time_data.device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.time_emb_dim, 2, device=seq_time_data.device).float()
            * (-math.log(10000.0) / self.time_emb_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0).repeat(batch_size, 1, 1)


class DCHL(nn.Module):
    def __init__(self, num_users, num_pois, args, device):
        super(DCHL, self).__init__()

        self.num_users = num_users
        self.num_pois = num_pois
        self.args = args
        self.device = device
        self.emb_dim = args.emb_dim
        self.ssl_temp = args.temperature

        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        self.poi_embedding = nn.Embedding(
            num_pois + 1, self.emb_dim, padding_idx=num_pois
        )

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.poi_embedding.weight)

        self.mv_hconv_network = MultiViewHyperConvNetwork(
            args.num_mv_layers, args.emb_dim, 0, device
        )
        self.geo_conv_network = GeoConvNetwork(args.num_geo_layers, args.dropout)
        self.di_hconv_network = DirectedHyperConvNetwork(
            args.num_di_layers, device, args.dropout
        )

        self.hyper_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.gcn_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.trans_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())

        self.user_hyper_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.user_gcn_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())

        self.pos_embeddings = nn.Embedding(1500, self.emb_dim, padding_idx=0)
        self.w_1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_dim, 1))
        self.glu1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.glu2 = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

        self.w_gate_geo = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_geo = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        self.w_gate_seq = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_seq = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        self.w_gate_col = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_col = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        nn.init.xavier_normal_(self.w_gate_geo.data)
        nn.init.xavier_normal_(self.b_gate_geo.data)
        nn.init.xavier_normal_(self.w_gate_seq.data)
        nn.init.xavier_normal_(self.b_gate_seq.data)
        nn.init.xavier_normal_(self.w_gate_col.data)
        nn.init.xavier_normal_(self.b_gate_col.data)

        self.w_gate_time = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_time = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        nn.init.xavier_normal_(self.w_gate_time.data)
        nn.init.xavier_normal_(self.b_gate_time.data)

        self.dropout = nn.Dropout(args.dropout)

        self.geo_gate_pois_embs = None
        self.seq_gate_pois_embs = None
        self.hg_structural_users_embs = None

        self.seq_model = SequenceModel(
            input_dim=args.emb_dim,
            hidden_dim=args.emb_dim,
            time_emb_dim=args.emb_dim,
        )
        self.temporal_attention_model = TemporalAttentionModel(
            user_emb_dim=args.emb_dim,
        )

    @staticmethod
    def row_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]

        return corrupted_embedding

    def cal_loss_infonce(self, emb1, emb2):
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.ssl_temp)
        neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.ssl_temp), axis=1)
        loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))

        loss /= pos_score.shape[0]

        return loss

    def cal_loss_cl_pois(self, hg_pois_embs, geo_pois_embs, trans_pois_embs):

        norm_hg_pois_embs = F.normalize(hg_pois_embs, p=2, dim=1)
        norm_geo_pois_embs = F.normalize(geo_pois_embs, p=2, dim=1)
        norm_trans_pois_embs = F.normalize(trans_pois_embs, p=2, dim=1)

        loss_cl_pois = 0.0
        loss_cl_pois += self.cal_loss_infonce(norm_hg_pois_embs, norm_geo_pois_embs)
        loss_cl_pois += self.cal_loss_infonce(norm_hg_pois_embs, norm_trans_pois_embs)
        loss_cl_pois += self.cal_loss_infonce(norm_geo_pois_embs, norm_trans_pois_embs)

        return loss_cl_pois

    def cal_loss_cl_users(
        self, hg_batch_users_embs, geo_batch_users_embs, trans_batch_users_embs
    ):

        norm_hg_batch_users_embs = F.normalize(hg_batch_users_embs, p=2, dim=1)
        norm_geo_batch_users_embs = F.normalize(geo_batch_users_embs, p=2, dim=1)
        norm_trans_batch_users_embs = F.normalize(trans_batch_users_embs, p=2, dim=1)

        loss_cl_users = 0.0
        loss_cl_users += self.cal_loss_infonce(
            norm_hg_batch_users_embs, norm_geo_batch_users_embs
        )
        loss_cl_users += self.cal_loss_infonce(
            norm_hg_batch_users_embs, norm_trans_batch_users_embs
        )
        loss_cl_users += self.cal_loss_infonce(
            norm_geo_batch_users_embs, norm_trans_batch_users_embs
        )

        return loss_cl_users

    def cal_loss_same_view(self, original_views, augmented_views):
        if len(original_views) != len(augmented_views):
            raise ValueError("Original and augmented views must have the same length")
        loss = original_views[0].new_tensor(0.0)
        for original_view, augmented_view in zip(original_views, augmented_views):
            norm_original_view = F.normalize(original_view, p=2, dim=1)
            norm_augmented_view = F.normalize(augmented_view, p=2, dim=1)
            loss = loss + self.cal_loss_infonce(norm_original_view, norm_augmented_view)
        return loss

    def combine_multiview_embeddings(self, user_views, poi_views):
        combined_user_embs = (
            self.hyper_gate(user_views[0]) * user_views[0]
            + self.gcn_gate(user_views[1]) * user_views[1]
            + self.trans_gate(user_views[2]) * user_views[2]
        )
        combined_poi_embs = poi_views[0] + poi_views[1] + poi_views[2]
        return combined_user_embs, combined_poi_embs

    def forward(self, dataset, batch):

        geo_gate_pois_embs = torch.multiply(
            self.poi_embedding.weight[:-1],
            torch.sigmoid(
                torch.matmul(self.poi_embedding.weight[:-1], self.w_gate_geo)
                + self.b_gate_geo
            ),
        )
        seq_gate_pois_embs = torch.multiply(
            self.poi_embedding.weight[:-1],
            torch.sigmoid(
                torch.matmul(self.poi_embedding.weight[:-1], self.w_gate_seq)
                + self.b_gate_seq
            ),
        )
        col_gate_pois_embs = torch.multiply(
            self.poi_embedding.weight[:-1],
            torch.sigmoid(
                torch.matmul(self.poi_embedding.weight[:-1], self.w_gate_col)
                + self.b_gate_col
            ),
        )

        hg_pois_embs = self.mv_hconv_network(
            col_gate_pois_embs,
            dataset.pad_all_train_sessions,
            dataset.HG_up,
            dataset.HG_pu,
        )

        hg_structural_users_embs = torch.sparse.mm(dataset.HG_up, hg_pois_embs)
        hg_batch_users_embs = hg_structural_users_embs[batch["user_idx"]]
        user_seq_embs = None

        if "user_time_seq" in batch:

            time_gate_pois_embs = torch.multiply(
                self.poi_embedding.weight[:-1],
                torch.sigmoid(
                    torch.matmul(self.poi_embedding.weight[:-1], self.w_gate_time)
                    + self.b_gate_time
                ),
            )
            zero_embedding = torch.zeros(
                1, self.emb_dim, device=time_gate_pois_embs.device
            )

            hg_pois_embs_cat = torch.cat([time_gate_pois_embs, zero_embedding], dim=0)

            user_time_seq = batch["user_time_seq"]
            user_seq_pois_embs = hg_pois_embs_cat[batch["user_seq"]]
            user_seq_embs = self.seq_model(user_seq_pois_embs, user_time_seq)
            hg_batch_users_embs = self.temporal_attention_model(
                hg_batch_users_embs, user_seq_embs
            )

        geo_pois_embs = self.geo_conv_network(geo_gate_pois_embs, dataset.poi_geo_graph)

        geo_structural_users_embs = torch.sparse.mm(dataset.HG_up, geo_pois_embs)
        geo_batch_users_embs = geo_structural_users_embs[batch["user_idx"]]

        trans_pois_embs = self.di_hconv_network(
            seq_gate_pois_embs, dataset.HG_poi_src, dataset.HG_poi_tar
        )

        trans_structural_users_embs = torch.sparse.mm(dataset.HG_up, trans_pois_embs)
        trans_batch_users_embs = trans_structural_users_embs[batch["user_idx"]]

        loss_cl_poi = self.cal_loss_cl_pois(
            hg_pois_embs, geo_pois_embs, trans_pois_embs
        )
        loss_cl_user = self.cal_loss_cl_users(
            hg_batch_users_embs, geo_batch_users_embs, trans_batch_users_embs
        )

        if self.training and all(
            hasattr(dataset, attr)
            for attr in (
                "HG_pu_aug",
                "HG_up_aug",
                "poi_geo_graph_aug",
                "HG_poi_src_aug",
                "HG_poi_tar_aug",
            )
        ):
            hg_pois_embs_aug = self.mv_hconv_network(
                col_gate_pois_embs,
                dataset.pad_all_train_sessions,
                dataset.HG_up_aug,
                dataset.HG_pu_aug,
            )
            hg_structural_users_embs_aug = torch.sparse.mm(
                dataset.HG_up_aug, hg_pois_embs_aug
            )
            hg_batch_users_embs_aug = hg_structural_users_embs_aug[batch["user_idx"]]
            if user_seq_embs is not None:
                hg_batch_users_embs_aug = self.temporal_attention_model(
                    hg_batch_users_embs_aug, user_seq_embs
                )

            geo_pois_embs_aug = self.geo_conv_network(
                geo_gate_pois_embs, dataset.poi_geo_graph_aug
            )
            geo_structural_users_embs_aug = torch.sparse.mm(
                dataset.HG_up, geo_pois_embs_aug
            )
            geo_batch_users_embs_aug = geo_structural_users_embs_aug[batch["user_idx"]]

            trans_pois_embs_aug = self.di_hconv_network(
                seq_gate_pois_embs,
                dataset.HG_poi_src_aug,
                dataset.HG_poi_tar_aug,
            )
            trans_structural_users_embs_aug = torch.sparse.mm(
                dataset.HG_up, trans_pois_embs_aug
            )
            trans_batch_users_embs_aug = trans_structural_users_embs_aug[
                batch["user_idx"]
            ]

            loss_cl_poi = loss_cl_poi + self.cal_loss_same_view(
                [hg_pois_embs, geo_pois_embs, trans_pois_embs],
                [hg_pois_embs_aug, geo_pois_embs_aug, trans_pois_embs_aug],
            )
            loss_cl_user = loss_cl_user + self.cal_loss_same_view(
                [hg_batch_users_embs, geo_batch_users_embs, trans_batch_users_embs],
                [
                    hg_batch_users_embs_aug,
                    geo_batch_users_embs_aug,
                    trans_batch_users_embs_aug,
                ],
            )

        norm_hg_pois_embs = F.normalize(hg_pois_embs, p=2, dim=1)
        norm_geo_pois_embs = F.normalize(geo_pois_embs, p=2, dim=1)
        norm_trans_pois_embs = F.normalize(trans_pois_embs, p=2, dim=1)

        norm_hg_batch_users_embs = F.normalize(hg_batch_users_embs, p=2, dim=1)
        norm_geo_batch_users_embs = F.normalize(geo_batch_users_embs, p=2, dim=1)
        norm_trans_batch_users_embs = F.normalize(trans_batch_users_embs, p=2, dim=1)

        combined_batch_users_embs, combined_pois_embs = (
            self.combine_multiview_embeddings(
                [
                    norm_hg_batch_users_embs,
                    norm_geo_batch_users_embs,
                    norm_trans_batch_users_embs,
                ],
                [
                    norm_hg_pois_embs,
                    norm_geo_pois_embs,
                    norm_trans_pois_embs,
                ],
            )
        )

        prediction = combined_batch_users_embs @ combined_pois_embs.T

        self.geo_gate_pois_embs = geo_gate_pois_embs
        self.seq_gate_pois_embs = seq_gate_pois_embs
        self.hg_structural_users_embs = hg_structural_users_embs

        return prediction, loss_cl_user, loss_cl_poi

    def get_poi_embedding(self):
        return self.poi_embedding.weight[:-1]

    def get_user_embedding(self):
        return self.user_embedding.weight

    def get_geo_poi_embedding(self):
        return self.geo_gate_pois_embs

    def get_seq_poi_embedding(self):
        return self.seq_gate_pois_embs

    def get_seq_user_embedding(self):
        return self.hg_structural_users_embs
