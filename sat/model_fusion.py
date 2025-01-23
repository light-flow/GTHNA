# -*- coding: utf-8 -*-
import torch
import math
import torch.nn.functional as F
from torch import nn
from experiments.utils import edge_index_to_adjacency
import torch_geometric.nn as gnn
from graphmae.models.edcoder import PreModel
from .layers import TransformerEncoderLayer
from einops import repeat


class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, edge_index, complete_edge_index,
                subgraph_node_index=None, subgraph_edge_index=None,
                subgraph_edge_attr=None, subgraph_indicator_index=None, edge_attr=None, degree=None,
                ptr=None, return_attn=False):
        output = x

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
        if self.norm is not None:
            output = self.norm(output)
        return output


class GraphTransformer(nn.Module):
    def __init__(self, in_size, num_class, d_model, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False, abs_pe=False, abs_pe_dim=0,
                 gnn_type="graph", se="gnn", use_edge_attr=False, num_edge_features=4,
                 in_embed=True, edge_embed=True, use_global_pool=True, max_seq_len=None,
                 global_pool='mean', **kwargs):
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

        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            edge_dim = kwargs.get('edge_dim', 32)
            if edge_embed:
                if isinstance(num_edge_features, int):
                    self.embedding_edge = nn.Embedding(num_edge_features, edge_dim)
                else:
                    raise ValueError("Not implemented!")
            else:
                self.embedding_edge = nn.Linear(in_features=num_edge_features,
                                                out_features=edge_dim, bias=False)
        else:
            kwargs['edge_dim'] = None

        self.gnn_type = gnn_type
        self.se = se
        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
            gnn_type=gnn_type, se=se, **kwargs)
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)
        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, d_model))
            self.pooling = None
        self.use_global_pool = use_global_pool

        self.max_seq_len = max_seq_len
        if max_seq_len is None:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(True),
                nn.Linear(d_model, num_class)
            )
        else:
            self.classifier = nn.ModuleList()
            for i in range(max_seq_len):
                self.classifier.append(nn.Linear(d_model, num_class))

    def forward(self, data, return_attn=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        node_depth = data.node_depth if hasattr(data, "node_depth") else None

        if self.se == "khopgnn":
            subgraph_node_index = data.subgraph_node_idx
            subgraph_edge_index = data.subgraph_edge_index
            subgraph_indicator_index = data.subgraph_indicator
            subgraph_edge_attr = data.subgraph_edge_attr if hasattr(data, "subgraph_edge_attr") \
                else None
        else:
            subgraph_node_index = None
            subgraph_edge_index = None
            subgraph_indicator_index = None
            subgraph_edge_attr = None

        complete_edge_index = data.complete_edge_index if hasattr(data, 'complete_edge_index') else None
        abs_pe = data.abs_pe if hasattr(data, 'abs_pe') else None
        degree = data.degree if hasattr(data, 'degree') else None
        output = self.embedding(x) if node_depth is None else self.embedding(x, node_depth.view(-1, ))

        if self.abs_pe and abs_pe is not None:
            abs_pe = self.embedding_abs_pe(abs_pe)
            output = output + abs_pe
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.embedding_edge(edge_attr)
            if subgraph_edge_attr is not None:
                subgraph_edge_attr = self.embedding_edge(subgraph_edge_attr)
        else:
            edge_attr = None
            subgraph_edge_attr = None

        if self.global_pool == 'cls' and self.use_global_pool:
            bsz = len(data.ptr) - 1
            if complete_edge_index is not None:
                new_index = torch.vstack((torch.arange(data.num_nodes).to(data.batch), data.batch + data.num_nodes))
                new_index2 = torch.vstack((new_index[1], new_index[0]))
                idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
                new_index3 = torch.vstack((idx_tmp, idx_tmp))
                complete_edge_index = torch.cat((
                    complete_edge_index, new_index, new_index2, new_index3), dim=-1)
            if subgraph_node_index is not None:
                idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
                subgraph_node_index = torch.hstack((subgraph_node_index, idx_tmp))
                subgraph_indicator_index = torch.hstack((subgraph_indicator_index, idx_tmp))
            degree = None
            cls_tokens = repeat(self.cls_token, '() d -> b d', b=bsz)
            output = torch.cat((output, cls_tokens))

        output = self.encoder(
            output,
            edge_index,
            complete_edge_index,
            edge_attr=edge_attr,
            degree=degree,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index,
            subgraph_edge_attr=subgraph_edge_attr,
            ptr=data.ptr,
            return_attn=return_attn
        )
        # readout step
        if self.use_global_pool:
            if self.global_pool == 'cls':
                output = output[-bsz:]
            else:
                output = self.pooling(output, data.batch)
        if self.max_seq_len is not None:
            pred_list = []
            for i in range(self.max_seq_len):
                pred_list.append(self.classifier[i](output))
            return pred_list
        return self.classifier(output)


class AdaptGraphTransformer(nn.Module):
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
        decoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
            gnn_type=gnn_type, se=se, **kwargs)
        self.decoder = GraphTransformerEncoder(decoder_layer, num_layers)
        self.de_embedding = nn.Linear(in_features=d_model,
                                      out_features=in_size,
                                      bias=False)
        self.max_seq_len = max_seq_len

    def forward(self, data, return_attn=False):
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
        # decoder step
        de_emd = self.decoder(
            out,
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
        X_pred = self.de_embedding(de_emd)
        A_pred = de_emd @ de_emd.T
        return X_pred, A_pred


class Attribute_Decoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Attribute_Decoder, self).__init__()

        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        return x


class Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout):
        super(Structure_Decoder, self).__init__()

        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T

        return x


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


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


class MemoryGraphTransformer(nn.Module):
    def __init__(self, in_size, d_model, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False, abs_pe=False, abs_pe_dim=0,
                 gnn_type="graph", se="gnn",
                 in_embed=True, max_seq_len=None,
                 **kwargs):
        super().__init__()

        self.abs_pe = abs_pe
        self.abs_pe_dim = abs_pe_dim
        self.fusion_attention = FusionAttentionLayer(feature_dim=d_model)
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

    def forward(self, data, memory, return_attn=False):
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
        output = self.embedding(x) if node_depth is None else self.embedding(x, node_depth.view(-1, ))

        if self.abs_pe and abs_pe is not None:
            abs_pe = self.embedding_abs_pe(abs_pe)
            # 根据节点度给节点位置编码权重
            # abs_pe = abs_pe * de_w.unsqueeze(dim=1)
            output = self.fusion_attention(abs_pe, output)

        # output = x_w * output
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
        output = read(memory, query=out)
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
        X_pred = torch.relu(self.de_embedding(de_emd))
        # de_emd = torch.sigmoid(de_emd)
        A_pred = de_emd @ de_emd.T
        return X_pred, A_pred, out


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
        X_pred = self.de_embedding(de_emd)
        A_pred = de_emd @ de_emd.T
        return X_pred, A_pred, out


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
