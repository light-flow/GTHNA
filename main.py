import argparse
import time

import torch
import torch.nn.functional as F
import numpy as np
import random
import os
from utils import transfer, loss_func, memory_loss, update_m, load_data, buildArgs
from sat.position_encoding import POSENCODINGS
from sat.gnn_layers import GNN_TYPES
from torch_geometric.loader import DataLoader
from torch import optim
from sat.data import GraphDataset
from sat.models import MemoryGraphTransformer
from sklearn.metrics import roc_auc_score


def load_args():
    parser = argparse.ArgumentParser(
        description='Structure-Aware Transformer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default="citeseer",
                        help='name of dataset')
    parser.add_argument('--num-heads', type=int, default=4, help="number of heads")
    parser.add_argument('--num-layers', type=int, default=2, help="number of layers")
    parser.add_argument('--dim-hidden', type=int, default=128, help="hidden dimension of Transformer")
    parser.add_argument('--dropout', type=float, default=0.0, help="dropout")
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--abs-pe', type=str, default='lap', choices=POSENCODINGS.keys(),
                        help='which absolute PE to use?')
    parser.add_argument('--abs-pe-dim', type=int, default=20, help='dimension for absolute PE')
    parser.add_argument('--gnn-type', type=str, default='graph',
                        choices=GNN_TYPES,
                        help="GNN structure extractor type")
    parser.add_argument('--k-hop', type=int, default=2, help="number of layers for GNNs")
    parser.add_argument('--se', type=str, default="gnn",
                        help='Extractor type: khopgnn, or gnn')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='weight of struct and attribute loss')
    parser.add_argument('--msize', type=int, default=512, help="number of memory items")
    parser.add_argument('--beta', type=float, default=0.8, help="normal ratio")
    parser.add_argument('--lambda_loss1', type=float, default=1e-2)  # neighbor reconstruction loss weight
    parser.add_argument('--lambda_loss2', type=float, default=0.5)  # feature loss weight
    parser.add_argument('--lambda_loss3', type=float, default=0.8)  # degree loss weight
    parser.add_argument('--sample_size', type=int, default=10)
    parser.add_argument('--s', type=float, default=1.0, help="node self reconstruct loss weight")
    parser.add_argument('--n', type=float, default=1e-3, help="node neighborhood reconstruct loss weight")
    parser.add_argument('--m', type=float, default=1e-3, help="node self reconstruct loss weight")
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    args = load_args()
    seed = args.seed
    dataset = args.dataset
    setup_seed(seed)
    print(f'Dataset {dataset} Run seed {seed}')
    data_name = dataset
    args = buildArgs(args, dataset)
    graph = load_data(data_name)
    # 节点属性归一化
    node_features = graph.x
    node_features_min = node_features.min()
    node_features_max = node_features.max()
    node_features = (node_features - node_features_min) / node_features_max
    graph.x = node_features
    # 计算节点邻居列表以及邻居数量
    in_nodes = graph.edge_index[0, :]
    out_nodes = graph.edge_index[1, :]

    neighbor_dict = {}
    for in_node, out_node in zip(in_nodes, out_nodes):
        if in_node.item() not in neighbor_dict:
            neighbor_dict[in_node.item()] = []
        neighbor_dict[in_node.item()].append(out_node.item())

    neighbor_num_list = []
    for i in neighbor_dict:
        neighbor_num_list.append(len(neighbor_dict[i]))
    neighbor_num_list = torch.tensor(neighbor_num_list).cuda()


    graph.y = graph.y.bool()
    X_laebl, A_label, y_label = transfer(graph)
    input_size = graph.x.shape[1]
    dataset = GraphDataset([graph],abs_pe_dim=args.abs_pe_dim, degree=True,
                           k_hop=args.k_hop, se=args.se, use_subgraph_edge_attr=False,
                           return_complete_index=False)
    loader = DataLoader(dataset, batch_size=1)
    final_data = next(iter(loader)).cuda()
    model = MemoryGraphTransformer(in_size=input_size,
                                   d_model=args.dim_hidden,
                                   dim_feedforward=2 * args.dim_hidden,
                                   dropout=args.dropout,
                                   num_heads=args.num_heads,
                                   num_layers=args.num_layers,
                                   abs_pe=args.abs_pe,
                                   abs_pe_dim=args.abs_pe_dim,
                                   gnn_type=args.gnn_type,
                                   k_hop=args.k_hop,
                                   se=args.se,
                                   lambda_loss1=args.lambda_loss1,
                                   lambda_loss2=args.lambda_loss2,
                                   lambda_loss3=args.lambda_loss3,
                                   neighbor_num_list=neighbor_num_list,
                                   sample_size=args.sample_size,
                                   dataset=data_name,
                                   in_embed=False).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    best_auc = 0
    # Initialize the memory items
    memory = F.normalize(torch.rand((args.msize, args.dim_hidden), dtype=torch.float),
                         dim=1).cuda()

    for epoch in range(args.epochs):
        model.train()
        X_pred, A_pred, loss, loss_per_node, node_embedding = model(final_data, memory, neighbor_num_list, neighbor_dict)
        loss_rec, struct_loss, feat_loss = loss_func(A_label, A_pred, X_laebl, X_pred, args.alpha)
        loss_com, loss_com_per_node, loss_separate = memory_loss(memory, node_embedding)
        s, n, m = args.s, args.n, args.m
        score = (n * loss_per_node + m * loss_com_per_node + s * loss_rec).detach().cpu().numpy()
        loss = n * loss + s * torch.mean(loss_rec) + m * (loss_separate + loss_com)
        auc = roc_auc_score(y_label.cpu(), score)
        if auc > best_auc:
            best_auc = auc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        memory = update_m(node_embedding.detach(), memory.detach(), loss_per_node.detach(), args.beta, True)

        print("Epoch:", '%04d' % (epoch + 1), "auc=", "{:.4f}".format(auc),"train_loss=", "{:.5f}".format(loss.item()))

        if (epoch + 1) % 10 == 0:
            model.eval()
            test_memory = memory.detach()

            X_pred, A_pred, loss, loss_per_node, node_embedding = model(final_data, test_memory, neighbor_num_list,
                                                                        neighbor_dict)
            loss_rec, struct_loss, feat_loss = loss_func(A_label, A_pred, X_laebl, X_pred, args.alpha)
            loss_com, loss_com_per_node, loss_separate = memory_loss(memory, node_embedding)
            s, n, m = args.s, args.n, args.m
            score = (n * loss_per_node + m * loss_com_per_node + s * loss_rec).detach().cpu().numpy()
            auc = roc_auc_score(y_label.cpu(), score)
            if auc > best_auc:
                best_auc = auc
            print("Epoch:", '%04d' % (epoch + 1), 'Auc', auc)
            print(f"Best_AUC: {best_auc}")

