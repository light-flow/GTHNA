import torch
# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric.data
import torch.nn.functional as F
import scipy.io as sio
from torch_geometric.data import Data


class ASTNodeEncoder(torch.nn.Module):
    '''
        Input:
            x: default node feature. the first and second column represents node type and node attributes.
            depth: The depth of the node in the AST.
        Output:
            emb_dim-dimensional vector
    '''

    def __init__(self, emb_dim, num_nodetypes, num_nodeattributes, max_depth):
        super(ASTNodeEncoder, self).__init__()

        self.max_depth = max_depth

        self.type_encoder = torch.nn.Embedding(num_nodetypes, emb_dim)
        self.attribute_encoder = torch.nn.Embedding(num_nodeattributes, emb_dim)
        self.depth_encoder = torch.nn.Embedding(self.max_depth + 1, emb_dim)

    def forward(self, x, depth):
        depth[depth > self.max_depth] = self.max_depth
        return self.type_encoder(x[:, 0]) + self.attribute_encoder(x[:, 1]) + self.depth_encoder(depth)


def get_vocab_mapping(seq_list, num_vocab):
    '''
        Input:
            seq_list: a list of sequences
            num_vocab: vocabulary size
        Output:
            vocab2idx:
                A dictionary that maps vocabulary into integer index.
                Additioanlly, we also index '__UNK__' and '__EOS__'
                '__UNK__' : out-of-vocabulary term
                '__EOS__' : end-of-sentence
            idx2vocab:
                A list that maps idx to actual vocabulary.
    '''

    vocab_cnt = {}
    vocab_list = []
    for seq in seq_list:
        for w in seq:
            if w in vocab_cnt:
                vocab_cnt[w] += 1
            else:
                vocab_cnt[w] = 1
                vocab_list.append(w)

    cnt_list = np.array([vocab_cnt[w] for w in vocab_list])
    topvocab = np.argsort(-cnt_list, kind='stable')[:num_vocab]

    print('Coverage of top {} vocabulary:'.format(num_vocab))
    print(float(np.sum(cnt_list[topvocab])) / np.sum(cnt_list))

    vocab2idx = {vocab_list[vocab_idx]: idx for idx, vocab_idx in enumerate(topvocab)}
    idx2vocab = [vocab_list[vocab_idx] for vocab_idx in topvocab]

    # print(topvocab)
    # print([vocab_list[v] for v in topvocab[:10]])
    # print([vocab_list[v] for v in topvocab[-10:]])

    vocab2idx['__UNK__'] = num_vocab
    idx2vocab.append('__UNK__')

    vocab2idx['__EOS__'] = num_vocab + 1
    idx2vocab.append('__EOS__')

    # test the correspondence between vocab2idx and idx2vocab
    for idx, vocab in enumerate(idx2vocab):
        assert (idx == vocab2idx[vocab])

    # test that the idx of '__EOS__' is len(idx2vocab) - 1.
    # This fact will be used in decode_arr_to_seq, when finding __EOS__
    assert (vocab2idx['__EOS__'] == len(idx2vocab) - 1)

    return vocab2idx, idx2vocab


def augment_edge(data):
    '''
        Input:
            data: PyG data object
        Output:
            data (edges are augmented in the following ways):
                data.edge_index: Added next-token edge. The inverse edges were also added.
                data.edge_attr (torch.Long):
                    data.edge_attr[:,0]: whether it is AST edge (0) for next-token edge (1)
                    data.edge_attr[:,1]: whether it is original direction (0) or inverse direction (1)
    '''

    ##### AST edge
    edge_index_ast = data.edge_index
    edge_attr_ast = torch.zeros((edge_index_ast.size(1), 2))

    ##### Inverse AST edge
    edge_index_ast_inverse = torch.stack([edge_index_ast[1], edge_index_ast[0]], dim=0)
    edge_attr_ast_inverse = torch.cat(
        [torch.zeros(edge_index_ast_inverse.size(1), 1), torch.ones(edge_index_ast_inverse.size(1), 1)], dim=1)

    ##### Next-token edge

    ## Obtain attributed nodes and get their indices in dfs order
    # attributed_node_idx = torch.where(data.node_is_attributed.view(-1,) == 1)[0]
    # attributed_node_idx_in_dfs_order = attributed_node_idx[torch.argsort(data.node_dfs_order[attributed_node_idx].view(-1,))]

    ## Since the nodes are already sorted in dfs ordering in our case, we can just do the following.
    attributed_node_idx_in_dfs_order = torch.where(data.node_is_attributed.view(-1, ) == 1)[0]

    ## build next token edge
    # Given: attributed_node_idx_in_dfs_order
    #        [1, 3, 4, 5, 8, 9, 12]
    # Output:
    #    [[1, 3, 4, 5, 8, 9]
    #     [3, 4, 5, 8, 9, 12]
    edge_index_nextoken = torch.stack([attributed_node_idx_in_dfs_order[:-1], attributed_node_idx_in_dfs_order[1:]],
                                      dim=0)
    edge_attr_nextoken = torch.cat(
        [torch.ones(edge_index_nextoken.size(1), 1), torch.zeros(edge_index_nextoken.size(1), 1)], dim=1)

    ##### Inverse next-token edge
    edge_index_nextoken_inverse = torch.stack([edge_index_nextoken[1], edge_index_nextoken[0]], dim=0)
    edge_attr_nextoken_inverse = torch.ones((edge_index_nextoken.size(1), 2))

    data.edge_index = torch.cat(
        [edge_index_ast, edge_index_ast_inverse, edge_index_nextoken, edge_index_nextoken_inverse], dim=1)
    data.edge_attr = torch.cat([edge_attr_ast, edge_attr_ast_inverse, edge_attr_nextoken, edge_attr_nextoken_inverse],
                               dim=0)

    return data


def encode_y_to_arr(data, vocab2idx, max_seq_len):
    '''
    Input:
        data: PyG graph object
        output: add y_arr to data 
    '''

    # PyG >= 1.5.0
    seq = data.y

    # PyG = 1.4.3
    # seq = data.y[0]

    data.y_arr = encode_seq_to_arr(seq, vocab2idx, max_seq_len)

    return data


def encode_seq_to_arr(seq, vocab2idx, max_seq_len):
    '''
    Input:
        seq: A list of words
        output: add y_arr (torch.Tensor)
    '''

    augmented_seq = seq[:max_seq_len] + ['__EOS__'] * max(0, max_seq_len - len(seq))
    return torch.tensor([[vocab2idx[w] if w in vocab2idx else vocab2idx['__UNK__'] for w in augmented_seq]],
                        dtype=torch.long)


def decode_arr_to_seq(arr, idx2vocab):
    '''
        Input: torch 1d array: y_arr
        Output: a sequence of words.
    '''

    eos_idx_list = torch.nonzero(arr == len(idx2vocab) - 1,
                                 as_tuple=False)  # find the position of __EOS__ (the last vocab in idx2vocab)
    if len(eos_idx_list) > 0:
        clippted_arr = arr[: torch.min(eos_idx_list)]  # find the smallest __EOS__
    else:
        clippted_arr = arr

    return list(map(lambda x: idx2vocab[x], clippted_arr.cpu()))


def edge_index_to_adjacency(edge_index, num_nodes):
    # 创建一个全零的邻接矩阵
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # 根据边索引列表更新邻接矩阵
    for edge in edge_index.T:
        adjacency_matrix[edge[0], edge[1]] = 1
        adjacency_matrix[edge[1], edge[0]] = 1  # 如果是有向图，可以去掉这行

    return torch.from_numpy(adjacency_matrix)


def transfer(graph: torch_geometric.data.Data):
    feat = graph.x
    adj = graph.edge_index
    anomaly_label = graph.y.to(torch.int)
    return feat.cuda(), edge_index_to_adjacency(adj, graph.num_nodes).cuda(), anomaly_label.cuda()


def loss_func(adj, A_hat, attrs, X_hat, alpha):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X_hat - attrs, 2)
    w1 = torch.where(diff_attribute == 0,
                     torch.tensor(1.0, dtype=diff_attribute.dtype).cuda(),
                     torch.tensor(2.0, dtype=diff_attribute.dtype).cuda())
    diff_attribute = w1 * diff_attribute
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    diff_structure = torch.pow(A_hat - adj, 2)
    w2 = torch.where(diff_structure == 0,
                     torch.tensor(1.0, dtype=diff_structure.dtype).cuda(),
                     torch.tensor(2.0, dtype=diff_structure.dtype).cuda())
    diff_structure = w2 * diff_structure
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost = alpha * attribute_reconstruction_errors + (1 - alpha) * structure_reconstruction_errors

    return cost, structure_cost, attribute_cost


def get_score(mem, query):
    score = torch.matmul(query, torch.t(mem))
    score_query = F.softmax(score, dim=0)
    score_memory = F.softmax(score, dim=1)
    return score_query, score_memory


def memory_loss(memory, node_embedding, margin=1.0):
    # # 计算每个特征向量与 memory 中所有向量的余弦相似度
    # similarity = torch.nn.functional.cosine_similarity(node_embedding.unsqueeze(1), memory.unsqueeze(0),
    #                                                    dim=2)  # 形状为 (n, m)
    #
    # # 找到每行的最近和第二最近的相似度
    # top_similarities, indices = torch.topk(similarity, 2, largest=True)
    #
    # # top_similarities[:, 0] 是最近的相似度
    # # top_similarities[:, 1] 是次近的相似度
    # first_nearest = top_similarities[:, 0].clone()
    # second_nearest = top_similarities[:, 1].clone()
    #
    # # 计算损失，使用 margin 来调整
    # separate_loss = torch.mean(torch.clamp((first_nearest - second_nearest), min=margin))
    ga_loss, ga_loss_per_node = gather_loss(node_embedding, memory)
    return ga_loss, ga_loss_per_node, spread_loss(node_embedding, memory, margin)


def spread_loss(query, keys, margin):
    loss = torch.nn.TripletMarginLoss(margin=margin)

    softmax_score_query, softmax_score_memory = get_score(keys, query)

    _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=1)

    # 1st, 2nd closest memories
    pos = keys[gathering_indices[:, 0]]
    neg = keys[gathering_indices[:, 1]]

    spreading_loss = loss(query, pos.detach(), neg.detach())

    return spreading_loss


def gather_loss(query, keys):
    loss_mse = torch.nn.MSELoss()

    softmax_score_query, softmax_score_memory = get_score(keys, query)

    _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)

    gathering_loss = loss_mse(query, keys[gathering_indices].squeeze(1).detach())
    gathering_loss_per_node = (query - keys[gathering_indices].squeeze(1).detach()).pow(2).mean(1)

    return gathering_loss, gathering_loss_per_node


def get_update_query(mem, max_indices, score, query, train):
    m, d = mem.size()
    if train:
        query_update = torch.zeros((m, d)).cuda()

        for i in range(m):
            idx = torch.nonzero(max_indices.squeeze(1) == i)
            a, _ = idx.size()
            if a != 0:
                query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)),
                                            dim=0)
            else:
                query_update[i] = 0

        return query_update

    else:
        query_update = torch.zeros((m, d)).cuda()
        for i in range(m):
            idx = torch.nonzero(max_indices.squeeze(1) == i)
            a, _ = idx.size()
            # ex = update_indices[0][i]
            if a != 0:
                # idx = idx[idx != ex]
                query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)),
                                            dim=0)
            #                     query_update[i] = torch.sum(query[idx].squeeze(1), dim=0)
            else:
                query_update[i] = 0

        return query_update


def update_m(query, keys, reconstruct_error, beta, train):
    threshold = torch.quantile(reconstruct_error, beta)
    mask = reconstruct_error < threshold
    filter_query = query[mask]
    softmax_score_query, softmax_score_memory = get_score(keys, filter_query)

    _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
    _, updating_indices = torch.topk(softmax_score_query, 1, dim=0)

    if train:
        # top-1 queries (of each memory) update (weighted sum) & random pick
        query_update = get_update_query(keys, gathering_indices, softmax_score_query,
                                        filter_query, train)
        updated_memory = F.normalize(query_update + keys, dim=1)

    else:
        # only weighted sum update when test
        query_update = get_update_query(keys, gathering_indices, softmax_score_query,
                                        filter_query, train)
        updated_memory = F.normalize(query_update + keys, dim=1)

        # top-1 update
        # query_update = query_reshape[updating_indices][0]
        # updated_memory = F.normalize(query_update + keys, dim=1)

    return updated_memory.detach()



def adj_to_edge(adj):
    num_nodes = len(adj)
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj[i][j] == 1:
                edges.append((i, j))
        edges.append((i, i))
    edge_index = torch.tensor(edges, dtype=torch.int64).t().contiguous()
    return edge_index



def load_data(data_name):
    mat_list = ['YelpHotel', 'YelpNYC', 'YelpRes', 'citeseer', 'pubmed', 'BlogCatalog', 'ACM', 'Amazon']
    if data_name in mat_list:
        data = sio.loadmat(f'./dataset/{data_name}.mat')
        # if data_name == 'Amazon':
        #     feat = data['features']
        #     adj = data['homo']
        #     label = data['label']
        # else:
        feat = data['Attributes']
        adj = data['Network']
        label = data['Label']

        label = label.flatten()
        feat = feat.toarray()
        adj = adj.toarray()
        edge_index = adj_to_edge(adj)
        graph = Data(x=torch.from_numpy(feat).float(), edge_index=edge_index, y=torch.from_numpy(label).float())
        return graph
    return torch.load(f'./dataset/{data_name}.pt')


# def visualization(labels, score, dataset_name):
#     nom_index = [i for i, label in enumerate(labels) if label == 0]
#     anom_index = [i for i, label in enumerate(labels) if label == 1]
#     # 可视化平均余弦相似度
#     plt.figure(figsize=(12, 6))
#
#     if nom_index:
#         plt.scatter(nom_index, [score[i] for i in nom_index],
#                     c='green', alpha=0.6, edgecolors='w', label='Normal')
#
#     if anom_index:
#         plt.scatter(anom_index, [score[i] for i in anom_index],
#                     c='red', alpha=0.6, edgecolors='w', label='Abnormal')
#
#     plt.title('Anomaly Score for Each Node with Its Neighbors')
#     plt.xlabel('Node Index')
#     plt.ylabel('Anomaly Score')
#     plt.legend()
#     plt.grid(True)
#     # 保存图片到本地
#     plt.savefig(f"{dataset_name}_score.png", dpi=300, bbox_inches='tight')


def buildArgs(args, dataset):
    if dataset == 'ACM':
        args.epochs = 200
    if dataset != 'inj_cora':
        args.se = 'gnn'
    if dataset in ['citeseer', 'inj_cora']:
        args.s = 1
        args.n = 1e-2
        args.m = 1e-2
        args.lr = 1e-2
    if dataset == 'weibo':
        args.s = 1
        args.n = 1e-7
        args.m = 1e-2
        args.lr = 1e-3
        args.alpha = 1
    if dataset == 'BlogCatalog':
        args.s = 1
        args.n = 1e-7
        args.m = 1e-2
        args.alpha = 0.95
        args.epochs = 600
    if dataset == 'ACM':
        args.s = 1
        args.n = 0
        args.m = 1e-2
        args.alpha = 0.9
        args.msize = 1024
        args.epochs = 200
    if dataset == 'YelpRes':
        args.s = 1
        args.n = 1e-7
        args.m = 1e-2
        args.alpha = 0.95
        args.epochs = 300
    if dataset == 'Amazon':
        args.s = 1
        args.n = 0
        args.m = 0
        args.alpha = 0.9
    return args


def buildArgsM(args, dataset):
    if dataset == 'ACM':
        args.epochs = 200
    if dataset != 'inj_cora':
        args.se = 'gnn'
    if dataset in ['citeseer', 'inj_cora']:
        args.s = 1
        args.n = 1e-2
        args.m = 1e-2
        args.lr = 1e-2
    if dataset == 'weibo':
        args.s = 1
        args.n = 1e-7
        args.m = 1e-2
        args.lr = 1e-3
        args.alpha = 1
    if dataset == 'BlogCatalog':
        args.s = 1
        args.n = 1e-7
        args.m = 1e-2
        args.alpha = 0.95
        args.epochs = 600
    if dataset == 'ACM':
        args.s = 1
        args.n = 0
        args.m = 1e-2
        args.alpha = 0.9
        args.msize = 1024
        args.epochs = 200
    if dataset == 'YelpRes':
        args.s = 1
        args.n = 1e-7
        args.m = 1e-2
        args.alpha = 0.95
        args.epochs = 300
    if dataset == 'Amazon':
        args.s = 1
        args.n = 0
        args.m = 0
        args.alpha = 0.9
    args.m = 0
    return args


def compute_gap(node_embeds, memory, label):
    node_embeds = node_embeds.detach()
    memory = memory.detach()
    label = label.squeeze()
    ano_emd = node_embeds[label]
    normal_emd = node_embeds[label == False]

    # L2 标准化（每个向量除以其范数）
    H_normal = normal_emd / normal_emd.norm(dim=1, keepdim=True)
    H_abnormal = ano_emd / ano_emd.norm(dim=1, keepdim=True)
    M = memory / memory.norm(dim=1, keepdim=True)

    # 计算内存项与正常节点之间的相似度
    cos_sim_normal = torch.matmul(M, H_normal.T)  # 计算内存项与正常节点的相似度 (m x q)
    max_sim_normal, _ = cos_sim_normal.max(dim=1)  # 每个内存项与正常节点的最大相似度

    # 计算内存项与异常节点之间的相似度
    cos_sim_abnormal = torch.matmul(M, H_abnormal.T)  # 计算内存项与异常节点的相似度 (m x p)
    min_sim_abnormal, _ = cos_sim_abnormal.min(dim=1)  # 每个内存项与异常节点的最大相似度
    return torch.mean(max_sim_normal - min_sim_abnormal)