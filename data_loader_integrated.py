import os
import numpy as np
import pandas as pd
import scipy.sparse as ssp
import scipy.io as sio
import torch
from torch_geometric.data import Data
import warnings

warnings.filterwarnings('ignore')


class IntegratedDataLoader:
    """整合的数据加载器，支持两种数据格式"""

    def __init__(self, dataset_name, k=10):
        """
        初始化数据加载器

        Parameters:
        -----------
        dataset_name : str
            数据集名称，如 'B-dataset', 'C-dataset', 'F-dataset'
        k : int
            Top-k过滤参数
        """
        self.dataset_name = dataset_name
        self.k = k
        self.dataset_path = f'./dataset/{dataset_name}'

    def load_from_csv(self):
        """从CSV文件加载数据并转换为SLGCN格式"""
        print(f"Loading dataset {self.dataset_name} from CSV files...")

        # 1. 读取原始CSV文件
        drug_drug = pd.read_csv(f'{self.dataset_path}/drug_drug.csv', header=None).values
        disease_disease = pd.read_csv(f'{self.dataset_path}/disease_disease.csv', header=None).values
        drug_disease = pd.read_csv(f'{self.dataset_path}/drug_disease.csv', header=None).values

        n_drug = drug_drug.shape[0]
        n_disease = disease_disease.shape[0]

        print(f"Drug: {n_drug}, Disease: {n_disease}")

        # 2. Top-k过滤
        def topk_filtering(sim_matrix, k):
            """Top-k过滤"""
            filtered = np.zeros_like(sim_matrix)
            for i in range(len(sim_matrix)):
                topk_idx = np.argpartition(sim_matrix[i], -k)[-k:]
                filtered[i, topk_idx] = sim_matrix[i, topk_idx]
            return filtered

        drug_drug_topk = topk_filtering(drug_drug, self.k)
        disease_disease_topk = topk_filtering(disease_disease, self.k)

        # 3. 构建异构邻接矩阵
        # 药物-药物相似度部分
        A_drug_drug = drug_drug_topk

        # 药物-疾病关联部分
        A_drug_disease = drug_disease

        # 疾病-疾病相似度部分
        A_disease_disease = disease_disease_topk

        # 4. 构建完整的邻接矩阵
        top_left = np.hstack([A_drug_drug, A_drug_disease])
        bottom_right = np.hstack([A_drug_disease.T, A_disease_disease])
        A = np.vstack([top_left, bottom_right])

        # 转换为稀疏矩阵
        A_sparse = ssp.csr_matrix(A)

        # 5. 构建特征矩阵（属性矩阵）
        # 药物特征 = [药物相似度, 药物-疾病关联]
        drug_feature = np.hstack([drug_drug, drug_disease])

        # 疾病特征 = [药物-疾病关联转置, 疾病相似度]
        disease_feature = np.hstack([drug_disease.T, disease_disease])

        attributes = np.vstack([drug_feature, disease_feature])

        # 6. 创建关联矩阵（用于负采样）
        net = A_sparse.copy()

        return {
            'net': net,  # 稀疏邻接矩阵
            'drug_sim': drug_drug,  # 药物相似度矩阵
            'disease_sim': disease_disease,  # 疾病相似度矩阵
            'didr': drug_disease.T,  # 药物-疾病关联矩阵转置（与旧格式一致）
            'attributes': attributes,  # 特征矩阵
            'drug_num': n_drug,
            'disease_num': n_disease,
            'A': A_sparse  # 完整的邻接矩阵
        }

    def load_from_mat(self):
        """从MAT文件加载数据（保持原SLGCN格式）"""
        print(f"Loading dataset {self.dataset_name} from MAT files...")

        # 读取关联矩阵
        data = sio.loadmat(f'data/{self.dataset_name}-A.mat')
        net = data['net']

        # 读取相似度信息
        old_data = sio.loadmat(f'data/{self.dataset_name}dataset.mat')
        drug_sim = old_data["drug"].astype(float)
        disease_sim = old_data["disease"].astype(float)
        drug_dis_matrix = old_data["didr"].T  # 注意转置

        drug_num = drug_sim.shape[0]
        disease_num = disease_sim.shape[0]

        # 处理相似度矩阵
        row1, col1 = np.diag_indices_from(drug_sim)
        drug_sim[row1, col1] = 0
        row2, col2 = np.diag_indices_from(disease_sim)
        disease_sim[row2, col2] = 0

        # 构建特征矩阵
        mat1 = np.hstack((drug_sim, drug_dis_matrix))
        mat2 = np.hstack((drug_dis_matrix.T, disease_sim))
        attributes = np.vstack((mat1, mat2))

        # 读取异构网络
        data_heter = sio.loadmat(f'data/{self.dataset_name}-1.mat')
        net_heter = data_heter['net']

        return {
            'net': net,
            'net_heter': net_heter,
            'drug_sim': drug_sim,
            'disease_sim': disease_sim,
            'didr': drug_dis_matrix,
            'attributes': attributes,
            'drug_num': drug_num,
            'disease_num': disease_num,
            'A': net_heter
        }

    def convert_to_heterogeneous_graph(self, data_dict):
        """将数据转换为异构图格式"""
        # 构建药物-药物邻接矩阵
        drug_drug_adj = data_dict['drug_sim']

        # 构建药物-疾病邻接矩阵
        drug_disease_adj = data_dict['didr']

        # 构建疾病-疾病邻接矩阵
        disease_disease_adj = data_dict['disease_sim']

        # 创建异构图数据
        hetero_data = {
            'drug_drug': drug_drug_adj,
            'drug_disease': drug_disease_adj,
            'disease_disease': disease_disease_adj,
            'drug_num': data_dict['drug_num'],
            'disease_num': data_dict['disease_num']
        }

        return hetero_data

    def prepare_for_slgcn(self, data_source='csv', apply_topk=True):
        """
        准备SLGCN所需的数据格式

        Parameters:
        -----------
        data_source : str
            数据源类型，'csv' 或 'mat'
        apply_topk : bool
            是否应用Top-k过滤
        """
        if data_source == 'csv':
            data = self.load_from_csv()
        else:
            data = self.load_from_mat()

        # 如果需要Top-k过滤且数据来自mat文件
        if apply_topk and data_source == 'mat':
            # 对相似度矩阵应用Top-k过滤
            drug_sim = data['drug_sim']
            disease_sim = data['disease_sim']

            def topk_filter(sim_matrix, k):
                filtered = np.zeros_like(sim_matrix)
                for i in range(len(sim_matrix)):
                    if np.sum(sim_matrix[i] > 0) > k:
                        topk_idx = np.argpartition(sim_matrix[i], -k)[-k:]
                        filtered[i, topk_idx] = sim_matrix[i, topk_idx]
                    else:
                        filtered[i] = sim_matrix[i]
                return filtered

            drug_sim_filtered = topk_filter(drug_sim, self.k)
            disease_sim_filtered = topk_filter(disease_sim, self.k)

            # 更新特征矩阵
            drug_num = data['drug_num']
            drug_dis_matrix = data['didr']

            mat1 = np.hstack((drug_sim_filtered, drug_dis_matrix))
            mat2 = np.hstack((drug_dis_matrix.T, disease_sim_filtered))
            data['attributes'] = np.vstack((mat1, mat2))

            # 更新相似度矩阵
            data['drug_sim'] = drug_sim_filtered
            data['disease_sim'] = disease_sim_filtered

        return data


def generate_meta_path_instances(data_dict, drug_id, disease_id, k=10):
    """
    生成元路径实例

    Parameters:
    -----------
    data_dict : dict
        包含网络数据
    drug_id : int
        药物ID
    disease_id : int
        疾病ID
    k : int
        邻居数量

    Returns:
    --------
    list
        元路径实例列表
    """
    n_drug = data_dict['drug_num']
    disease_offset = n_drug  # 疾病节点的偏移量

    # 转换疾病ID为全局节点ID
    disease_global_id = disease_offset + disease_id

    # 获取药物邻居
    drug_sim = data_dict['drug_sim']
    drug_neighbors = np.where(drug_sim[drug_id] > 0)[0]
    if len(drug_neighbors) > k:
        drug_neighbors = np.random.choice(drug_neighbors, k, replace=False)

    # 获取疾病邻居
    disease_sim = data_dict['disease_sim']
    disease_neighbors = np.where(disease_sim[disease_id] > 0)[0]
    disease_neighbors_global = disease_offset + disease_neighbors
    if len(disease_neighbors) > k:
        disease_neighbors = np.random.choice(disease_neighbors, k, replace=False)
        disease_neighbors_global = disease_offset + disease_neighbors

    # 生成元路径实例
    meta_paths = []

    # 类型1: 直接路径 (drug -> disease)
    meta_paths.append([drug_id, disease_global_id])

    # 类型2: 通过药物邻居 (drug -> drug_neighbor -> disease)
    for neighbor in drug_neighbors:
        meta_paths.append([drug_id, neighbor, disease_global_id])

    # 类型3: 通过疾病邻居 (drug -> disease -> disease_neighbor)
    for neighbor in disease_neighbors_global:
        meta_paths.append([drug_id, disease_global_id, neighbor])

    # 类型4: 通过药物和疾病邻居 (drug -> drug_neighbor -> disease -> disease_neighbor)
    for drug_n in drug_neighbors[:min(3, len(drug_neighbors))]:
        for disease_n in disease_neighbors_global[:min(3, len(disease_neighbors_global))]:
            meta_paths.append([drug_id, drug_n, disease_global_id, disease_n])

    return meta_paths


def create_subgraph_from_meta_path(meta_path, data_dict, hop=2):
    """
    从元路径创建子图

    Parameters:
    -----------
    meta_path : list
        元路径节点序列
    data_dict : dict
        包含网络数据
    hop : int
        子图跳数

    Returns:
    --------
    Data
        PyG格式的图数据
    """
    # 获取邻接矩阵
    A = data_dict['A']

    # 提取元路径中的节点
    nodes = set(meta_path)

    # 扩展hop跳邻居
    current_nodes = set(nodes)
    for _ in range(hop):
        new_nodes = set()
        for node in current_nodes:
            neighbors = A[node].indices
            new_nodes.update(neighbors)
        nodes.update(new_nodes)
        current_nodes = new_nodes

    nodes = list(nodes)
    node_dict = {node: idx for idx, node in enumerate(nodes)}

    # 提取子图邻接关系
    edge_list = []
    for i, node_i in enumerate(nodes):
        neighbors = A[node_i].indices
        for node_j in neighbors:
            if node_j in node_dict:
                edge_list.append([i, node_dict[node_j]])

    if len(edge_list) == 0:
        # 如果没有边，添加自环
        edge_list = [[i, i] for i in range(len(nodes))]

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # 提取节点特征
    attributes = data_dict['attributes']
    x = torch.tensor(attributes[nodes], dtype=torch.float32)

    # 创建图数据
    graph_data = Data(x=x, edge_index=edge_index)

    return graph_data


# 修改主函数中的train部分
def train_with_integrated_data(args):
    """使用整合数据的主训练函数"""
    # 初始化数据加载器
    data_loader = IntegratedDataLoader(args.data_name, k=args.k)

    # 选择数据源
    if args.data_source == 'csv':
        data_dict = data_loader.prepare_for_slgcn(data_source='csv', apply_topk=True)
    else:
        data_dict = data_loader.prepare_for_slgcn(data_source='mat', apply_topk=args.apply_topk)

    # 准备数据
    net = data_dict['net']
    attributes = data_dict['attributes']
    drug_num = data_dict['drug_num']
    disease_num = data_dict['disease_num']

    # 加载异构网络（如果存在）
    if 'net_heter' in data_dict:
        net_heter = data_dict['net_heter']
    else:
        net_heter = data_dict['A']

    # 创建输出目录
    os.makedirs('runtime', exist_ok=True)

    # 保存配置
    config_dict = vars(args)
    with open('runtime/config.txt', 'w') as f:
        for key, value in config_dict.items():
            f.write(f"{key}: {value}\n")

    # 10折交叉验证
    fold_results = []

    for fold_id, (train_pos, train_neg, test_pos, test_neg) in enumerate(
            sample_neg(net, drug_num, disease_num, args.pro, args.test_ratio,
                       max_train_num=args.max_train_num)
    ):
        if fold_id >= 10:
            break

        print(f"\n{'=' * 60}")
        print(f"Processing Fold {fold_id + 1}/10")
        print(f"{'=' * 60}")

        # 准备数据
        A = net_heter.copy()
        A[test_pos[0], test_pos[1]] = 0
        A[test_pos[1], test_pos[0]] = 0
        A.eliminate_zeros()

        # 训练和评估
        metrics = SLGCN(args, fold_id, A, train_pos, train_neg,
                        test_pos, test_neg, attributes)
        fold_results.append(metrics)

        gc.collect()

    # 输出结果（与原代码相同）
    print("\n" + "=" * 80)
    print("10-Fold Cross Validation Results - Enhanced SLGCN")
    print("=" * 80)

    all_metrics = np.array(fold_results)
    for i, metrics in enumerate(all_metrics):
        print(f"Fold {i + 1:2d}: Loss: {metrics[0]:.4f}, Acc: {metrics[1]:.4f}, "
              f"AUC: {metrics[2]:.4f}, AUPR: {metrics[3]:.4f}")

    # 统计信息
    mean_metrics = np.mean(all_metrics, axis=0)
    std_metrics = np.std(all_metrics, axis=0)

    print("\n" + "-" * 80)
    print("Summary Statistics:")
    print("-" * 80)
    print(f"Loss:     {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
    print(f"Accuracy: {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
    print(f"AUC:      {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
    print(f"AUPR:     {mean_metrics[3]:.4f} ± {std_metrics[3]:.4f}")

    # 保存结果
    results_dict = {
        'fold_results': all_metrics,
        'mean': mean_metrics,
        'std': std_metrics,
        'config': vars(args)
    }

    np.save('runtime/results.npy', results_dict)

    print(f"\nResults saved to: runtime/results.npy")
    print("=" * 80)