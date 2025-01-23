# -*- coding: utf-8 -*-
import torch
import math
import torch.nn.functional as F
from torch import nn
from graphmae.models.edcoder import PreModel
from .layers import TransformerEncoderLayer
from scipy.linalg import sqrtm
from torch_geometric.nn import SAGEConv, PNAConv
import random


class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, edge_index, complete_edge_index,
                subgraph_node_index=None, subgraph_edge_index=None,
                subgraph_edge_attr=None, subgraph_indicator_index=None, edge_attr=None, degree=None,
                ptr=None, return_attn=False):
        output = x
        out_list = []
        for mod in self.layers:
            output = mod(output, edge_index, complete_edge_index,
                         edge_attr=edge_attr, degree=degree,
                         subgraph_node_index=subgraph_node_index,
                         subgraph_edge_index=subgraph_edge_index,
                         subgraph_indicator_index=subgraph_indicator_index,
                         subgraph_edge_attr=subgraph_edge_attr,
                         ptr=ptr,
                         return_attn=return_attn
                         )
            out_list.append(output)
        if self.norm is not None:
            output = self.norm(output)
        loss_layer = GraphTransformerEncoder.layer_loss(out_list)
        return output, loss_layer

    @staticmethod
    def layer_loss(layer_outs):
        loss = 0
        for i in range(len(layer_outs) - 1):
            cos_sim = F.cosine_similarity(layer_outs[i], layer_outs[i + 1], dim=1)
            loss += 1 - cos_sim.mean()
        return loss



def get_score(mem, query):
    score = torch.matmul(query, torch.t(mem))
    score_query = F.softmax(score, dim=0)
    score_memory = F.softmax(score, dim=1)
    return score_query, score_memory


def read(mem, query):
    score_query, score_memory = get_score(mem.clone(), query.clone())
    read_memory = torch.matmul(score_memory.clone(), mem.clone())
    read_embedding = torch.cat((query.clone(), read_memory), dim=1)
    return read_embedding


def update(mem, query, reconstruct_error, beta):
    threshold = torch.quantile(reconstruct_error, beta)
    mask = reconstruct_error < threshold
    filter_query = query[mask]
    score_query, score_memory = get_score(mem, filter_query)
    update_mem = mem + torch.matmul(score_query.T, filter_query)
    new_memory = F.normalize(update_mem, dim=1)
    return new_memory


def KL_neighbor_loss(predictions, targets, mask_len):
    x1 = predictions.squeeze().cpu().detach()
    x2 = targets.squeeze().cpu().detach()

    mean_x1 = x1.mean(0)
    mean_x2 = x2.mean(0)

    nn = x1.shape[0]
    h_dim = x1.shape[1]

    cov_x1 = (x1 - mean_x1).transpose(1, 0).matmul(x1 - mean_x1) / max((nn - 1), 1)
    cov_x2 = (x2 - mean_x2).transpose(1, 0).matmul(x2 - mean_x2) / max((nn - 1), 1)

    eye = torch.eye(h_dim)
    cov_x1 = cov_x1 + eye
    cov_x2 = cov_x2 + eye

    KL_loss = 0.5 * (math.log(torch.det(cov_x1) / torch.det(cov_x2)) - h_dim + torch.trace(
        torch.inverse(cov_x2).matmul(cov_x1))
                     + (mean_x2 - mean_x1).reshape(1, -1).matmul(torch.inverse(cov_x2)).matmul(mean_x2 - mean_x1))
    KL_loss = KL_loss.cuda()
    return KL_loss


def W2_neighbor_loss(predictions, targets, mask_len):
    x1 = predictions.squeeze().cpu().detach()
    x2 = targets.squeeze().cpu().detach()

    mean_x1 = x1.mean(0)
    mean_x2 = x2.mean(0)

    nn = x1.shape[0]

    cov_x1 = (x1 - mean_x1).transpose(1, 0).matmul(x1 - mean_x1) / (nn - 1)
    cov_x2 = (x2 - mean_x2).transpose(1, 0).matmul(x2 - mean_x2) / (nn - 1)

    W2_loss = torch.square(mean_x1 - mean_x2).sum() + torch.trace(cov_x1 + cov_x2
                                                                  + 2 * sqrtm(
        sqrtm(cov_x1) @ (cov_x2.numpy()) @ (sqrtm(cov_x1))))

    return W2_loss


class MemoryGraphTransformer(nn.Module):
    def __init__(self, in_size, d_model, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False, abs_pe=False, abs_pe_dim=0,
                 gnn_type="graph", se="gnn",
                 in_embed=True, max_seq_len=None,
                 sample_size=5,
                 lambda_loss1=0,
                 lambda_loss2=0,
                 lambda_loss3=0,
                 neighbor_num_list=None,
                 dataset=None,
                 **kwargs):
        super().__init__()

        self.abs_pe = abs_pe
        self.abs_pe_dim = abs_pe_dim
        self.fusion_attention = FusionAttentionLayer(feature_dim=d_model)
        self.degree_decoder = FNN(d_model * 2, d_model * 2, 1, 4)
        # self.x_weight = nn.Linear(in_size, 1)

        if abs_pe and abs_pe_dim > 0:
            self.embedding_abs_pe = nn.Linear(abs_pe_dim, d_model)
        if in_embed:
            if isinstance(in_size, int):
                self.embedding = nn.Embedding(in_size, d_model)
            elif isinstance(in_size, nn.Module):
                self.embedding = in_size
            else:
                raise ValueError("Not implemented!")
        else:
            self.embedding = nn.Linear(in_features=in_size,
                                       out_features=d_model,
                                       bias=False)
        self.gnn_type = gnn_type
        self.dataset = dataset
        self.se = se
        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
            gnn_type=gnn_type, se=se, **kwargs)
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)
        decoder_layer = TransformerEncoderLayer(
            d_model * 2, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
            gnn_type=gnn_type, se=se, **kwargs)
        self.decoder = GraphTransformerEncoder(decoder_layer, num_layers)
        self.de_embedding = nn.Linear(in_features=d_model * 2,
                                      out_features=in_size,
                                      bias=False)
        self.max_seq_len = max_seq_len
        self.degree_loss_func = nn.MSELoss()
        self.feature_decoder = FNN(d_model * 2, d_model * 2, d_model, 3)
        self.feature_loss_func = nn.MSELoss()
        self.layer1_generator = MLP_generator(d_model * 2, d_model, sample_size)
        self.sample_size = sample_size
        self.out_dim = d_model

        # self.pool = mp.Pool(4)
        hidden_dim = d_model * 2
        self.mlp_gaussian_mean = nn.Parameter(
            torch.FloatTensor(hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim)).cuda()
        self.mlp_gaussian_log_sigma = nn.Parameter(
            torch.FloatTensor(hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim)).cuda()
        self.mlp_m = torch.distributions.Normal(torch.zeros(hidden_dim), torch.ones(hidden_dim))
        self.m = torch.distributions.Normal(torch.zeros(sample_size, hidden_dim),
                                            torch.ones(sample_size, hidden_dim))
        self.mlp_mean = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_sigma = nn.Linear(hidden_dim, hidden_dim)
        self.lambda_loss1 = lambda_loss1
        self.lambda_loss2 = lambda_loss2
        self.lambda_loss3 = lambda_loss3

        self.neighbor_num_list = neighbor_num_list
        self.tot_node = len(neighbor_num_list)

        self.mean_agg = SAGEConv(d_model, d_model, aggr="mean", normalize = False)
        # self.mean_agg = GraphSAGE(hidden_dim, hidden_dim, aggr='mean', num_layers=1)
        self.std_agg = PNAConv(d_model, d_model, aggregators=["std"],scalers=["identity"], deg=neighbor_num_list)
        self.m_batched = torch.distributions.Normal(torch.zeros(sample_size, self.tot_node, hidden_dim),
                                            torch.ones(sample_size, self.tot_node, hidden_dim))

    def degree_decoding(self, node_embeddings):
        degree_logits = F.relu(self.degree_decoder(node_embeddings))
        return degree_logits

    def neighbor_decoder(self, gij, ground_truth_degree_matrix, h0, neighbor_dict, edge_index, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        # Degree decoder below: gij 卷积得到的特征，h0图卷积之前的初步映射
        tot_nodes = gij.shape[0]
        degree_logits = self.degree_decoding(gij)
        ground_truth_degree_matrix = torch.unsqueeze(ground_truth_degree_matrix, dim=1)

        degree_loss = self.degree_loss_func(degree_logits, ground_truth_degree_matrix.float())
        degree_loss_per_node = (degree_logits - ground_truth_degree_matrix).pow(2)
        _, degree_masks = torch.max(degree_logits.data, dim=1)
        h_loss = 0
        feature_loss = 0
        # layer 1
        loss_list = []
        loss_list_per_node = []
        feature_loss_list = []
        # Sample multiple times to remove noise
        for _ in range(1):
            local_index_loss_sum = 0
            local_index_loss_sum_per_node = []
            indexes = list(range(tot_nodes))
            h0_prime = self.feature_decoder(gij)
            # feature_losses = self.feature_loss_func(h0, h0_prime)
            feature_losses_per_node = (h0 - h0_prime).pow(2).mean(1)
            feature_loss_list.append(feature_losses_per_node)

            # for i1, embedding in enumerate(gij):
            #     indexes.append(i1)
            # local_index_loss, local_index_loss_per_node = self.reconstruction_neighbors(self.layer1_generator, indexes,
            #                                                                             neighbor_dict, gij, h0, device)
            local_index_loss, local_index_loss_per_node = self.reconstruction_neighbors2(gij, h0, edge_index, device)

            loss_list.append(local_index_loss)
            loss_list_per_node.append(local_index_loss_per_node)

        loss_list = torch.stack(loss_list)
        h_loss += torch.mean(loss_list)

        loss_list_per_node = torch.stack(loss_list_per_node)
        h_loss_per_node = torch.mean(loss_list_per_node, dim=0)

        feature_loss_per_node = torch.mean(torch.stack(feature_loss_list), dim=0)
        feature_loss += torch.mean(torch.stack(feature_loss_list))

        h_loss_per_node = h_loss_per_node.reshape(tot_nodes, 1)
        degree_loss_per_node = degree_loss_per_node.reshape(tot_nodes, 1)
        feature_loss_per_node = feature_loss_per_node.reshape(tot_nodes, 1)

        loss = self.lambda_loss1 * h_loss + degree_loss * self.lambda_loss3 + self.lambda_loss2 * feature_loss
        loss_per_node = self.lambda_loss1 * h_loss_per_node + degree_loss_per_node * self.lambda_loss3 + self.lambda_loss2 * feature_loss_per_node

        return loss, loss_per_node, h_loss_per_node, degree_loss_per_node, feature_loss_per_node

    def sample_neighbors(self, indexes, neighbor_dict, gt_embeddings):
        sampled_embeddings_list = []
        mark_len_list = []
        for index in indexes:
            sampled_embeddings = []
            neighbor_indexes = neighbor_dict[index]
            if len(neighbor_indexes) < self.sample_size:
                mask_len = len(neighbor_indexes)
                sample_indexes = neighbor_indexes
            else:
                sample_indexes = random.sample(neighbor_indexes, self.sample_size)
                mask_len = self.sample_size
            for index in sample_indexes:
                sampled_embeddings.append(gt_embeddings[index].tolist())
            if len(sampled_embeddings) < self.sample_size:
                for _ in range(self.sample_size - len(sampled_embeddings)):
                    sampled_embeddings.append(torch.zeros(self.out_dim).tolist())
            sampled_embeddings_list.append(sampled_embeddings)
            mark_len_list.append(mask_len)

        return sampled_embeddings_list, mark_len_list

    def reconstruction_neighbors2(self, l1, h0, edge_index, device):

        recon_loss = 0
        recon_loss_per_node = []

        sample_sz_per_node = [self.sample_size] * self.tot_node

        # mean_neigh = self.graphconv1(h0, edge_index)
        mean_neigh = self.mean_agg(h0, edge_index).detach()
        std_neigh = self.std_agg(h0, edge_index).detach()
        # mean_neigh = self.graphconv2(mean_neigh, edge_index)
        # mean_neigh = l1
        # mean_neigh = self.mean_agg(h0, edge_index, num_sampled_nodes_per_hop=sample_sz_per_node)

        cov_neigh = torch.bmm(std_neigh.unsqueeze(dim=-1), std_neigh.unsqueeze(dim=1))

        target_mean = mean_neigh
        target_cov = cov_neigh

        self_embedding = l1
        # self_embedding = _normalize(self_embedding)

        self_embedding = self_embedding.unsqueeze(0)
        self_embedding = self_embedding.repeat(self.sample_size, 1, 1)
        generated_mean = self.mlp_mean(self_embedding)
        generated_sigma = self.mlp_sigma(self_embedding)

        std_z = self.m_batched.sample().to(device)
        var = generated_mean + generated_sigma.exp() * std_z
        nhij = self.layer1_generator(var)

        generated_mean = torch.mean(nhij, dim=0)
        generated_std = torch.std(nhij, dim=0)
        generated_cov = torch.bmm(generated_std.unsqueeze(dim=-1), generated_std.unsqueeze(dim=1)) / self.sample_size

        tot_nodes = l1.shape[0]
        h_dim = int(l1.shape[1] / 2)

        single_eye = torch.eye(h_dim).to(device)
        single_eye = single_eye.unsqueeze(dim=0)
        batch_eye = single_eye.repeat(tot_nodes, 1, 1)

        target_cov = target_cov + batch_eye
        generated_cov = generated_cov + batch_eye

        det_target_cov = torch.linalg.det(target_cov)
        det_generated_cov = torch.linalg.det(generated_cov)
        trace_mat = torch.matmul(torch.inverse(generated_cov), target_cov)

        x = torch.bmm(torch.unsqueeze(generated_mean - target_mean, dim=1), torch.inverse(generated_cov))
        y = torch.unsqueeze(generated_mean - target_mean, dim=-1)
        z = torch.bmm(x, y).squeeze()

        KL_loss = 0.5 * (torch.log(det_target_cov / det_generated_cov) - h_dim + trace_mat.diagonal(offset=0, dim1=-1,
                                                                                                    dim2=-2).sum(
            -1) + z)

        recon_loss = torch.mean(KL_loss)
        recon_loss_per_node = KL_loss

        return recon_loss, recon_loss_per_node

    def reconstruction_neighbors(self, FNN_generator, neighbor_indexes, neighbor_dict, from_layer, to_layer, device):
        # from_layer 卷积得到的信息 gil ，to_layer: h0
        local_index_loss = 0
        local_index_loss_per_node = []
        sampled_embeddings_list, mark_len_list = self.sample_neighbors(neighbor_indexes, neighbor_dict, to_layer)
        for i, neighbor_embeddings1 in enumerate(sampled_embeddings_list):
            # Generating h^k_v, reparameterization trick
            index = neighbor_indexes[i]
            mask_len1 = mark_len_list[i]
            mean = from_layer[index].repeat(self.sample_size, 1)
            mean = self.mlp_mean(mean)
            sigma = from_layer[index].repeat(self.sample_size, 1)
            sigma = self.mlp_sigma(sigma)
            std_z = self.m.sample().to(device)
            var = mean + sigma.exp() * std_z
            nhij = FNN_generator(var)

            generated_neighbors = nhij
            sum_neighbor_norm = 0

            for indexi, generated_neighbor in enumerate(generated_neighbors):
                sum_neighbor_norm += torch.norm(generated_neighbor) / math.sqrt(self.out_dim)
            generated_neighbors = torch.unsqueeze(generated_neighbors, dim=0).to(device)
            target_neighbors = torch.unsqueeze(torch.FloatTensor(neighbor_embeddings1), dim=0).to(device)



            KL_loss = KL_neighbor_loss(generated_neighbors, target_neighbors, mask_len1)
            local_index_loss += KL_loss
            local_index_loss_per_node.append(KL_loss)

        local_index_loss_per_node = torch.stack(local_index_loss_per_node)
        return local_index_loss, local_index_loss_per_node



    def forward(self, data, memory, ground_truth_degree_matrix, neighbor_dict, return_attn=False):
        x, edge_index, edge_attr, de_w = data.x, data.edge_index, data.edge_attr, data.degree
        # x_w = self.x_weight(x)
        node_depth = data.node_depth if hasattr(data, "node_depth") else None

        if self.se == "khopgnn":
            subgraph_node_index = data.subgraph_node_idx
            subgraph_edge_index = data.subgraph_edge_index
            subgraph_indicator_index = data.subgraph_indicator
        else:
            subgraph_node_index = None
            subgraph_edge_index = None
            subgraph_indicator_index = None

        complete_edge_index = data.complete_edge_index if hasattr(data, 'complete_edge_index') else None
        abs_pe = data.abs_pe if hasattr(data, 'abs_pe') else None
        h0 = self.embedding(x) if node_depth is None else self.embedding(x, node_depth.view(-1, ))

        if self.abs_pe and abs_pe is not None:
            abs_pe = self.embedding_abs_pe(abs_pe)
            # 根据节点度给节点位置编码权重
            # abs_pe = abs_pe * de_w.unsqueeze(dim=1)
            # output = torch.cat((output, abs_pe), dim=1)
            output = self.fusion_attention(h0, abs_pe)
        out, loss_layer_en = self.encoder(
            output,
            edge_index,
            complete_edge_index,
            edge_attr=None,
            degree=None,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index,
            subgraph_edge_attr=None,
            ptr=data.ptr,
            return_attn=return_attn
        )
        # read memory
        output = read(memory, query=out)
        de_emd, loss_layer_de = self.decoder(
            output,
            edge_index,
            complete_edge_index,
            edge_attr=None,
            degree=None,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index,
            subgraph_edge_attr=None,
            ptr=data.ptr,
            return_attn=return_attn)
        # reconstruct graph
        X_pred = self.de_embedding(de_emd)
        if self.dataset in ['ACM', 'YelpRes', 'Amazon']:
            A_pred = torch.sigmoid(de_emd) @ de_emd.T
        else:
            A_pred = de_emd @ de_emd.T
        loss, loss_per_node, h_loss_per_node, degree_loss_per_node, feature_loss_per_node = self.neighbor_decoder(
            gij=output,
            ground_truth_degree_matrix=ground_truth_degree_matrix,
            h0=h0,
            neighbor_dict=neighbor_dict,
            edge_index=edge_index
        )
        # return X_pred, A_pred, out, loss_layer_en, loss_layer_de
        return X_pred, A_pred, loss, loss_per_node.squeeze(), out


class SplitMemoryGraphTransformer(nn.Module):
    def __init__(self, in_size, d_model, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False, abs_pe=False, abs_pe_dim=0,
                 gnn_type="graph", se="gnn",
                 in_embed=True, max_seq_len=None,
                 **kwargs):
        super().__init__()

        self.abs_pe = abs_pe
        self.abs_pe_dim = abs_pe_dim
        if abs_pe and abs_pe_dim > 0:
            self.embedding_abs_pe = nn.Linear(abs_pe_dim, d_model)
        if in_embed:
            if isinstance(in_size, int):
                self.embedding = nn.Embedding(in_size, d_model)
            elif isinstance(in_size, nn.Module):
                self.embedding = in_size
            else:
                raise ValueError("Not implemented!")
        else:
            self.embedding = nn.Linear(in_features=in_size,
                                       out_features=d_model,
                                       bias=False)
        self.gnn_type = gnn_type
        self.se = se
        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
            gnn_type=gnn_type, se=se, **kwargs)
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)
        # decoder_layer = TransformerEncoderLayer(
        #     d_model * 2, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
        #     gnn_type=gnn_type, se=se, **kwargs)
        # self.decoder_attr = GraphTransformerEncoder(decoder_layer, num_layers)
        # self.de_embedding = nn.Linear(in_features=d_model * 2,
        #                               out_features=in_size,
        #                               bias=False)
        self.attr_decoder = PreModel()
        self.max_seq_len = max_seq_len

    def forward(self, data, memory_attr, memory_struct, return_attn=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        node_depth = data.node_depth if hasattr(data, "node_depth") else None

        if self.se == "khopgnn":
            subgraph_node_index = data.subgraph_node_idx
            subgraph_edge_index = data.subgraph_edge_index
            subgraph_indicator_index = data.subgraph_indicator
        else:
            subgraph_node_index = None
            subgraph_edge_index = None
            subgraph_indicator_index = None

        complete_edge_index = data.complete_edge_index if hasattr(data, 'complete_edge_index') else None
        abs_pe = data.abs_pe if hasattr(data, 'abs_pe') else None
        output = self.embedding(x) if node_depth is None else self.embedding(x, node_depth.view(-1, ))

        if self.abs_pe and abs_pe is not None:
            abs_pe = self.embedding_abs_pe(abs_pe)
            output = output + abs_pe

        out = self.encoder(
            output,
            edge_index,
            complete_edge_index,
            edge_attr=None,
            degree=None,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index,
            subgraph_edge_attr=None,
            ptr=data.ptr,
            return_attn=return_attn
        )
        # read memory
        output_attr = read(memory_attr, query=out)
        output_struct = read(memory_struct, query=out)

        de_emd = self.decoder(
            output,
            edge_index,
            complete_edge_index,
            edge_attr=None,
            degree=None,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index,
            subgraph_edge_attr=None,
            ptr=data.ptr,
            return_attn=return_attn)
        # reconstruct graph
        X_pred = torch.nn.PReLU(self.de_embedding(de_emd))
        A_pred = de_emd @ de_emd.T
        return X_pred, A_pred, out


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


# FNN
class FNN(nn.Module):
    def __init__(self, in_features, hidden, out_features, layer_num):
        super(FNN, self).__init__()
        self.linear1 = MLP(layer_num, in_features, hidden, out_features)
        self.linear2 = nn.Linear(out_features, out_features)

    def forward(self, embedding):
        x = self.linear1(embedding)
        x = self.linear2(F.relu(x))
        return x


class MLP_generator(nn.Module):
    def __init__(self, input_dim, output_dim, sample_size):
        super(MLP_generator, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.linear3 = nn.Linear(output_dim, output_dim)
        self.linear4 = nn.Linear(output_dim, output_dim)

    def forward(self, embedding):
        neighbor_embedding = F.relu(self.linear(embedding))
        neighbor_embedding = F.relu(self.linear2(neighbor_embedding))
        neighbor_embedding = F.relu(self.linear3(neighbor_embedding))
        neighbor_embedding = self.linear4(neighbor_embedding)
        return neighbor_embedding



class FusionAttentionLayer(nn.Module):
    def __init__(self, feature_dim):
        super(FusionAttentionLayer, self).__init__()
        # 定义融合注意力的线性变换层
        self.attention_fc = nn.Linear(feature_dim * 2, 1)

    def forward(self, h_struct, h_attr):
        h_concat = torch.cat((h_struct, h_attr), dim=1)
        alpha = torch.sigmoid(self.attention_fc(h_concat))
        h_fusion = alpha * h_struct + (1 - alpha) * h_attr
        return h_fusion