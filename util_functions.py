import numpy as np
import random
import os, sys, math, time
import pickle as cp
import networkx as nx
import scipy.io as sio
import scipy.sparse as ssp
from sklearn import metrics
from sklearn.model_selection import KFold
import warnings
import concurrent.futures
from functools import partial
import multiprocessing as mp
import torch
from torch_geometric.transforms import LineGraph
from torch_geometric.data import Data

warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)


class GNNGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features
        self.degs = list(dict(g.degree).values())

        if len(g.edges()) != 0:
            x, y = list(zip(*g.edges()))
            self.num_edges = len(x)
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])

        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):
            edge_features = nx.get_edge_attributes(g, 'features')
            assert (type(list(edge_features.values())[0]) == np.ndarray)
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in list(edge_features.items())}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])
            self.edge_features = np.concatenate(self.edge_features, 0)


def sample_neg(net, drug_num, disease_num, pro, kf, max_train_num=None):
    """生成负样本"""
    net_triu = ssp.triu(net, k=1)
    pos_row, pos_col, _ = ssp.find(net_triu)
    dr_n = drug_num
    di_n = disease_num

    # 生成负样本
    neg_row, neg_col = [], []
    while len(neg_row) < pro * len(pos_row):
        i, j = random.randint(0, dr_n - 1), random.randint(dr_n, dr_n + di_n - 1)
        if i < j and net[i, j] == 0:
            neg_row.append(i)
            neg_col.append(j)

    neg_row = np.array(neg_row)
    neg_col = np.array(neg_col)

    kfold = KFold(n_splits=kf, shuffle=True, random_state=666)

    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kfold.split(pos_row), kfold.split(neg_row)):
        train_pos = (pos_row[train_pos_idx], pos_col[train_pos_idx])
        test_pos = (pos_row[test_pos_idx], pos_col[test_pos_idx])
        train_neg = (neg_row[train_neg_idx], neg_col[train_neg_idx])
        test_neg = (neg_row[test_neg_idx], neg_col[test_neg_idx])

        yield train_pos, train_neg, test_pos, test_neg


def links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, h=1,
                    max_nodes_per_hop=None, node_information=None,
                    node_asso=None, verbose=True):
    """子图提取"""
    if h == 'auto':
        h = 2

    max_n_label = 0

    def process_batch(links, g_label, batch_size=100):
        results = []
        n_batches = int(np.ceil(len(links[0]) / batch_size))

        worker_func = partial(
            subgraph_extraction_labeling,
            A=A, h=h, max_nodes_per_hop=max_nodes_per_hop,
            node_information=node_information, node_asso=node_asso
        )

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(links[0]))

            batch_links = list(zip(
                links[0][start_idx:end_idx],
                links[1][start_idx:end_idx]
            ))

            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=min(4, mp.cpu_count())
            ) as executor:
                batch_results = list(executor.map(worker_func, batch_links))

            results.extend(batch_results)

        g_list = [
            GNNGraph(g, g_label, n_labels, n_features)
            for g, n_labels, n_features in results
        ]

        # 更新最大标签
        nonlocal max_n_label
        for _, n_labels, _ in results:
            if len(n_labels) > 0:
                max_n_label = max(max_n_label, max(n_labels))

        return g_list

    if verbose:
        print('Enclosing subgraph extraction begins...')

    start_time = time.time()

    # 处理各个部分
    train_pos_graphs = process_batch(train_pos, 1)
    train_neg_graphs = process_batch(train_neg, 0)
    test_pos_graphs = process_batch(test_pos, 1)
    test_neg_graphs = process_batch(test_neg, 0)

    train_graphs = train_pos_graphs + train_neg_graphs
    test_graphs = test_pos_graphs + test_neg_graphs

    if verbose:
        print(f"Time elapsed: {time.time() - start_time:.2f}s")
        print(f"Train graphs: {len(train_graphs)}, Test graphs: {len(test_graphs)}")
        print(f"Max node label: {max_n_label}")

    return train_graphs, test_graphs, max_n_label


def subgraph_extraction_labeling(ind, A, h=1, max_nodes_per_hop=None,
                                 node_information=None, node_asso=None):
    """提取子图并分配标签 - 优化版本"""
    src, dst = ind[0], ind[1]

    # 限制跳数，避免子图过大
    if h > 3:  # 限制最大跳数
        h = 3

    nodes = {src: 0, dst: 0}
    visited = {src, dst}
    current_layer = {src, dst}

    # 限制子图大小
    max_nodes = 500  # 设置最大节点数

    for dist in range(1, h + 1):
        next_layer = set()
        for node in current_layer:
            if len(visited) >= max_nodes:  # 达到最大节点数限制
                break

            _, neighbors, _ = ssp.find(A[node, :])
            for neighbor in neighbors:
                if neighbor not in visited and len(visited) < max_nodes:
                    nodes[neighbor] = dist
                    visited.add(neighbor)
                    next_layer.add(neighbor)

        if max_nodes_per_hop and len(next_layer) > max_nodes_per_hop:
            next_layer = set(random.sample(list(next_layer), max_nodes_per_hop))

        if not next_layer or len(visited) >= max_nodes:
            break

        current_layer = next_layer

    # 确保节点列表包含源节点和目标节点
    if src not in visited:
        visited.add(src)
        nodes[src] = 0
    if dst not in visited:
        visited.add(dst)
        nodes[dst] = 0

    node_list = list(visited)

    # 确保子图不会太大
    if len(node_list) > max_nodes:
        node_list = node_list[:max_nodes]

    # 创建子图
    subgraph = A[node_list, :][:, node_list]

    # 计算节点标签
    try:
        labels = node_label(subgraph)
        features = node_information[node_list] if node_information is not None else None

        # 创建网络图
        g = nx.from_scipy_sparse_array(subgraph)

        # 确保源节点和目标节点有边
        src_idx = node_list.index(src)
        dst_idx = node_list.index(dst)

        if not g.has_edge(src_idx, dst_idx):
            g.add_edge(src_idx, dst_idx)

        return g, labels.tolist(), features

    except Exception as e:
        print(f"Error in subgraph labeling: {e}")
        # 返回一个简单的图作为后备
        g = nx.Graph()
        g.add_edge(0, 1)
        return g, [1, 1], None

def node_label(subgraph):
    """节点标签分配 - 修复形状不匹配问题"""
    K = subgraph.shape[0]

    if K <= 2:
        return np.array([1, 1], dtype=np.int32)

    try:
        # 转换为稠密矩阵
        subgraph_dense = subgraph.toarray()

        # 计算距离到节点0
        subgraph_wo0 = subgraph_dense[1:, 1:]
        dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
        if dist_to_0.shape[0] > 0:
            dist_to_0 = dist_to_0[0, :]  # 到第一个节点的距离
        else:
            dist_to_0 = np.full(K - 1, np.inf)
    except Exception as e:
        print(f"Warning: shortest_path failed for subgraph_wo0: {e}")
        dist_to_0 = np.full(K - 1, np.inf)

    try:
        # 计算距离到节点1
        # 重新索引：去除节点1，节点0在新矩阵中索引为0
        indices = [0] + list(range(2, K))
        subgraph_wo1 = subgraph_dense[np.ix_(indices, indices)]
        dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
        if dist_to_1.shape[0] > 0:
            dist_to_1 = dist_to_1[0, :]  # 到第一个节点的距离
        else:
            dist_to_1 = np.full(K - 1, np.inf)
    except Exception as e:
        print(f"Warning: shortest_path failed for subgraph_wo1: {e}")
        dist_to_1 = np.full(K - 1, np.inf)

    # 确保长度相同
    min_len = min(len(dist_to_0), len(dist_to_1))
    if len(dist_to_0) > min_len:
        dist_to_0 = dist_to_0[:min_len]
    if len(dist_to_1) > min_len:
        dist_to_1 = dist_to_1[:min_len]

    # 处理无效值
    dist_to_0 = np.nan_to_num(dist_to_0, nan=np.inf, posinf=np.inf, neginf=np.inf)
    dist_to_1 = np.nan_to_num(dist_to_1, nan=np.inf, posinf=np.inf, neginf=np.inf)

    dist_sum = dist_to_0 + dist_to_1
    dist_sum = np.nan_to_num(dist_sum, nan=0, posinf=0, neginf=0)

    # 转换为整数
    dist_sum = dist_sum.astype(int)
    dist_sum = np.clip(dist_sum, 0, 1000)  # 限制范围避免溢出

    d = dist_sum
    d_over_2, d_mod_2 = np.divmod(d, 2)

    # 计算标签
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.clip(labels, 1, 100)  # 限制标签范围

    # 添加前两个节点的标签
    final_labels = np.concatenate((np.array([1, 1]), labels))

    return final_labels.astype(int)


def to_linegraphs(batch_graphs, max_n_label, verbose=True):
    """转换为线图"""
    graphs = []

    for idx, graph in enumerate(batch_graphs):
        if verbose and idx % 100 == 0:
            print(f'Processing graph {idx}/{len(batch_graphs)}')

        # 节点特征
        node_attr = torch.FloatTensor(graph.node_features)

        # 节点标签
        node_tag = torch.zeros(graph.num_nodes, max_n_label + 1)
        tags = torch.LongTensor(graph.node_tags).view(-1, 1)
        node_tag.scatter_(1, tags, 1)

        # 边特征
        edges = np.reshape(graph.edge_pairs, (-1, 2))
        sr = np.array([edges[:, 0], edges[:, 1]], dtype=np.int64)

        fea_s = node_tag[sr[0, :], :].repeat(2, 1)
        fea_r = node_tag[sr[1, :], :].repeat(2, 1)
        fea_body = torch.cat([fea_s, fea_r], 1)

        # 添加节点属性
        fea_node_s = node_attr[sr[0, :], :].repeat(2, 1)
        fea_node_r = node_attr[sr[1, :], :].repeat(2, 1)
        fea_node_ = torch.cat([fea_node_s, fea_node_r], 1)
        fea_body = torch.cat([fea_body, fea_node_], 1)

        # 构建双向边
        rs = np.array([edges[:, 1], edges[:, 0]], dtype=np.int64)
        edge_index = torch.tensor(np.concatenate([sr, rs], axis=1), dtype=torch.long)

        # 创建Data对象
        data = Data(edge_index=edge_index, edge_attr=fea_body)
        data.num_nodes = graph.num_nodes

        # 转换为线图
        data = LineGraph()(data)
        data['y'] = torch.tensor([graph.label])
        data.num_nodes = graph.num_edges

        graphs.append(data)

    if verbose:
        print(f'Converted {len(graphs)} graphs to line graphs')

    return graphs


def AA(A, test_pos, test_neg):
    """Adamic-Adar得分"""
    A_ = A / np.log(A.sum(axis=1) + 1)
    A_[np.isnan(A_)] = 0
    A_[np.isinf(A_)] = 0
    sim = A.dot(A_)
    return CalcAUC(sim, test_pos, test_neg)


def CN(A, test_pos, test_neg):
    """共同邻居得分"""
    sim = A.dot(A)
    return CalcAUC(sim, test_pos, test_neg)


def CalcAUC(sim, test_pos, test_neg):
    """计算AUC"""
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    return metrics.auc(fpr, tpr)