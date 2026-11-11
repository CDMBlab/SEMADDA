import torch
import csv
from torch.utils.data.sampler import *
import numpy as np
import sys, copy, math, time, pdb
import pickle as pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
import gc
import time
from util_functions import *
from torch_geometric.data import DataLoader
from model import *
from torch.utils.data import TensorDataset
from sklearn import metrics
from sklearn.metrics import average_precision_score
import warnings
import multiprocessing as mp
import pandas as pd  # 新增导入

warnings.filterwarnings('ignore')

# 强制使用CPU
device = torch.device('cpu')


# ====================== 整合的数据加载器 ======================
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

        # 根据您的项目结构构建路径
        # 项目根目录: D:\PyCharm\pythonProject2\一月三日复现出来的模型\SLGCN-main
        # 数据集路径: D:\PyCharm\pythonProject2\一月三日复现出来的模型\SLGCN-main\dataset\{dataset_name}
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"当前文件位置: {current_dir}")

        # 尝试几种可能的路径
        possible_paths = [
            os.path.join(current_dir, 'dataset', dataset_name),  # Main.py在根目录下
            os.path.join(current_dir, '..', 'dataset', dataset_name),  # Main.py在子目录下
            os.path.join(current_dir, '..', '..', 'dataset', dataset_name),  # Main.py在更深目录
            f'D:/PyCharm/pythonProject2/一月三日复现出来的模型/SLGCN-main/dataset/{dataset_name}',  # 绝对路径
        ]

        for path in possible_paths:
            path = os.path.abspath(path)
            if os.path.exists(path):
                self.dataset_path = path
                print(f"找到数据集路径: {self.dataset_path}")
                break
        else:
            # 如果都没找到，列出当前目录结构
            print("当前工作目录:", os.getcwd())
            print("当前目录下的内容:", os.listdir('.'))
            if os.path.exists('dataset'):
                print("dataset目录下的内容:", os.listdir('dataset'))
            raise FileNotFoundError(f"找不到数据集目录: {dataset_name}")

        # 列出数据集目录下的文件
        print(f"数据集目录下的文件:")
        for file in os.listdir(self.dataset_path):
            if file.endswith('.csv'):
                print(f"  - {file}")

    def load_from_csv(self):
        """从CSV文件加载数据并转换为SLGCN格式"""
        print(f"\n从CSV文件加载数据集 {self.dataset_name}...")

        # 列出所有CSV文件以便调试
        csv_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.csv')]
        print(f"可用的CSV文件: {csv_files}")

        # 尝试读取各种可能的文件名模式
        def find_file(pattern_list):
            """查找匹配模式的文件"""
            for pattern in pattern_list:
                for file in csv_files:
                    if pattern.lower() in file.lower():
                        file_path = os.path.join(self.dataset_path, file)
                        print(f"读取文件: {file}")
                        return pd.read_csv(file_path, header=None).values
            raise FileNotFoundError(f"未找到匹配 {pattern_list} 的文件")

        # 药物-药物相似度矩阵（尝试多种可能的名字）
        try:
            drug_drug = find_file(['drug_drug', 'drug_sim', 'drug'])
            print(f"药物相似度矩阵形状: {drug_drug.shape}")
        except FileNotFoundError:
            # 如果找不到，尝试其他可能性
            print("警告: 未找到药物相似度矩阵文件")
            # 列出所有CSV文件让用户选择
            for i, file in enumerate(csv_files):
                print(f"{i}: {file}")
            # 这里可以添加交互式选择，暂时退出
            raise

        # 疾病-疾病相似度矩阵
        try:
            disease_disease = find_file(['disease_disease', 'disease_sim', 'disease'])
            print(f"疾病相似度矩阵形状: {disease_disease.shape}")
        except FileNotFoundError:
            print("警告: 未找到疾病相似度矩阵文件")
            raise

        # 药物-疾病关联矩阵
        try:
            drug_disease = find_file(['drug_disease', 'association', 'didr', 'interaction'])
            print(f"药物-疾病关联矩阵形状: {drug_disease.shape}")
        except FileNotFoundError:
            print("警告: 未找到药物-疾病关联矩阵文件")
            # 如果没有关联矩阵，创建一个全0矩阵
            n_drug = drug_drug.shape[0]
            n_disease = disease_disease.shape[0]
            drug_disease = np.zeros((n_drug, n_disease))
            print(f"创建全零关联矩阵: {drug_disease.shape}")

        n_drug = drug_drug.shape[0]
        n_disease = disease_disease.shape[0]

        print(f"\n数据统计:")
        print(f"药物数量: {n_drug}")
        print(f"疾病数量: {n_disease}")
        print(f"关联矩阵中已知关联数: {np.sum(drug_disease > 0)}")

        # Top-k过滤函数
        def topk_filtering(sim_matrix, k):
            """Top-k过滤，只保留每行最大的k个值"""
            filtered = np.zeros_like(sim_matrix)
            for i in range(len(sim_matrix)):
                # 获取相似度最高的k个邻居（排除自己）
                row = sim_matrix[i].copy()
                # 如果是相似度矩阵，对角线通常是自己，设为最小值
                if i < len(row):
                    row[i] = -np.inf if i < n_drug else -np.inf

                # 找到top-k索引
                topk_indices = np.argpartition(row, -k)[-k:]
                filtered[i, topk_indices] = sim_matrix[i, topk_indices]
            return filtered

        print(f"\n应用Top-{self.k}过滤...")
        drug_drug_topk = topk_filtering(drug_drug, self.k)
        disease_disease_topk = topk_filtering(disease_disease, self.k)

        # 构建完整的邻接矩阵
        print("构建异构邻接矩阵...")
        # 药物-药物相似度部分
        A_drug_drug = drug_drug_topk

        # 药物-疾病关联部分
        A_drug_disease = drug_disease

        # 疾病-疾病相似度部分
        A_disease_disease = disease_disease_topk

        # 构建完整的邻接矩阵
        top_left = np.hstack([A_drug_drug, A_drug_disease])
        bottom_right = np.hstack([A_drug_disease.T, A_disease_disease])
        A = np.vstack([top_left, bottom_right])

        # 转换为稀疏矩阵
        A_sparse = ssp.csr_matrix(A)

        # 构建特征矩阵（属性矩阵）
        print("构建特征矩阵...")
        # 药物特征 = [药物相似度, 药物-疾病关联]
        drug_feature = np.hstack([drug_drug, drug_disease])

        # 疾病特征 = [药物-疾病关联转置, 疾病相似度]
        disease_feature = np.hstack([drug_disease.T, disease_disease])

        attributes = np.vstack([drug_feature, disease_feature])

        # 创建关联矩阵（用于负采样）
        net = A_sparse.copy()

        # 将药物-疾病关联矩阵转换为MATLAB格式的didr（转置）
        didr_mat = drug_disease.T

        print(f"\n构建完成:")
        print(f"完整邻接矩阵形状: {A.shape}")
        print(f"特征矩阵形状: {attributes.shape}")
        print(f"药物-疾病关联矩阵(didr)形状: {didr_mat.shape}")

        return {
            'net': net,  # 稀疏邻接矩阵
            'drug_sim': drug_drug,  # 原始药物相似度矩阵
            'disease_sim': disease_disease,  # 原始疾病相似度矩阵
            'didr': didr_mat,  # 药物-疾病关联矩阵转置（与旧格式一致）
            'attributes': attributes,  # 特征矩阵
            'drug_num': n_drug,  # 药物数量
            'disease_num': n_disease,  # 疾病数量
            'A': A_sparse,  # 完整的邻接矩阵（稀疏）
            'drug_drug_topk': drug_drug_topk,  # Top-k过滤后的药物相似度
            'disease_disease_topk': disease_disease_topk  # Top-k过滤后的疾病相似度
        }

    def load_from_mat(self):
        """从MAT文件加载数据（保持原SLGCN格式）"""
        print(f"从MAT文件加载数据集 {self.dataset_name}...")

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

    def prepare_for_slgcn(self, data_source='csv', apply_topk=True):
        """
        准备SLGCN所需的数据格式

        Parameters:
        -----------
        data_source : str
            数据源类型，'csv' 或 'mat'
        apply_topk : bool
            是否应用Top-k过滤（仅对mat数据有效）
        """
        if data_source == 'csv':
            print(f"使用CSV数据源: {self.dataset_name}")
            data = self.load_from_csv()
        else:
            print(f"使用MAT数据源: {self.dataset_name}")
            data = self.load_from_mat()

        # 如果需要Top-k过滤且数据来自mat文件
        if apply_topk and data_source == 'mat':
            print(f"对MAT数据应用Top-{self.k}过滤...")
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


# ====================== 参数解析函数 ======================
def parse(print_help=False):
    parser = argparse.ArgumentParser(description='Link Prediction with Enhanced GNN')
    # general settings
    parser.add_argument('--data-name', default='B-dataset', help='dataset name')
    parser.add_argument('--train-name', default=None, help='train name')
    parser.add_argument('--test-name', default=None, help='test name')
    parser.add_argument('--max-train-num', type=int, default=10000,
                        help='set maximum number of train links')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--test-ratio', type=int, default=10,
                        help='ratio of test links')
    # model settings
    parser.add_argument('--hop', default=2, help='enclosing subgraph hop number')
    parser.add_argument('--max-nodes-per-hop', default=100,
                        help='upper bound the # nodes per hop by subsampling')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning ratio')
    parser.add_argument('--dropout-rate', type=float, default=0.4, help='dropout rate')
    parser.add_argument('--epoch', type=int, default=30, help='epoches')
    parser.add_argument('--simw', type=int, default=2, help='simw')
    parser.add_argument('--layer', type=int, default=3, help='layer')
    parser.add_argument('--hidden', type=int, default=128, help='hidden')
    parser.add_argument('--pro', type=int, default=1, help='probability of negative and positive samples')
    # 新增参数
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--n-workers', type=int, default=4,
                        help='number of workers for parallel processing')
    # 数据源参数
    parser.add_argument('--data-source', type=str, default='csv',
                        choices=['csv', 'mat'], help='data source type')
    parser.add_argument('--k', type=int, default=10,
                        help='top-k parameter for similarity filtering')
    parser.add_argument('--apply-topk', action='store_true', default=True,
                        help='apply top-k filtering to similarity matrices')
    # 模型选择参数
    parser.add_argument('--model-type', type=str, default='enhanced',
                        choices=['original', 'hierarchical', 'enhanced'],
                        help='model type')
    parser.add_argument('--use-hierarchy', action='store_true', default=False,
                        help='use hierarchical line graph GNN')
    # 增强版特定参数
    parser.add_argument('--use-multi-scale', action='store_true', default=True,
                        help='use multi-scale GNN layers')
    parser.add_argument('--num-heads', type=int, default=4,
                        help='number of attention heads')
    # 稳健改进参数
    parser.add_argument('--use-layer-norm', action='store_true', default=True,
                        help='use layer normalization')
    parser.add_argument('--use-residual', action='store_true', default=True,
                        help='use residual connections')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='label smoothing parameter')
    parser.add_argument('--use-cosine-scheduler', action='store_true', default=True,
                        help='use cosine annealing learning rate scheduler')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='gradient clipping norm')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='warmup epochs for learning rate')
    # 训练策略
    parser.add_argument('--grad-accumulation', type=int, default=4,
                        help='gradient accumulation steps')
    parser.add_argument('--patience', type=int, default=10,
                        help='early stopping patience')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if print_help:
        parser.print_help()
    return args


# ====================== 训练函数 ======================
def train_with_integrated_data(args):
    """使用整合数据的主训练函数"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.hop != 'auto':
        args.hop = int(args.hop)
    if args.max_nodes_per_hop is not None:
        args.max_nodes_per_hop = int(args.max_nodes_per_hop)

    # 初始化数据加载器
    print(f"\n{'=' * 60}")
    print(f"初始化数据加载器...")
    print(f"数据集: {args.data_name}")
    print(f"数据源: {args.data_source}")
    print(f"Top-k参数: {args.k}")
    print(f"{'=' * 60}")

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
    runtime_dir = 'runtime'
    os.makedirs(runtime_dir, exist_ok=True)

    # 保存配置
    config_dict = vars(args)
    with open(f'{runtime_dir}/config.txt', 'w') as f:
        for key, value in config_dict.items():
            f.write(f"{key}: {value}\n")

    # 10折交叉验证
    fold_results = []

    print(f"\n开始10折交叉验证...")
    for fold_id, (train_pos, train_neg, test_pos, test_neg) in enumerate(
            sample_neg(net, drug_num, disease_num, args.pro, args.test_ratio,
                       max_train_num=args.max_train_num)
    ):
        if fold_id >= 10:
            break

        print(f"\n{'=' * 60}")
        print(f"处理第 {fold_id + 1}/10 折")
        print(f"{'=' * 60}")

        # 准备数据
        A = net_heter.copy()
        # 在测试集中移除边
        A[test_pos[0], test_pos[1]] = 0
        A[test_pos[1], test_pos[0]] = 0
        A.eliminate_zeros()

        # 训练和评估
        metrics_result = SLGCN(args, fold_id, A, train_pos, train_neg,
                               test_pos, test_neg, attributes)
        fold_results.append(metrics_result)

        gc.collect()

    # 输出结果
    print("\n" + "=" * 80)
    print("10折交叉验证结果 - 增强版 SLGCN")
    print("=" * 80)

    all_metrics = np.array(fold_results)
    for i, metrics_result in enumerate(all_metrics):
        print(f"第 {i + 1:2d} 折: 损失: {metrics_result[0]:.4f}, 准确率: {metrics_result[1]:.4f}, "
              f"AUC: {metrics_result[2]:.4f}, AUPR: {metrics_result[3]:.4f}")

    # 统计信息
    mean_metrics = np.mean(all_metrics, axis=0)
    std_metrics = np.std(all_metrics, axis=0)

    print("\n" + "-" * 80)
    print("统计摘要:")
    print("-" * 80)
    print(f"损失:     {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
    print(f"准确率: {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
    print(f"AUC:      {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
    print(f"AUPR:     {mean_metrics[3]:.4f} ± {std_metrics[3]:.4f}")

    # 保存结果
    results_dict = {
        'fold_results': all_metrics,
        'mean': mean_metrics,
        'std': std_metrics,
        'config': vars(args)
    }

    np.save(f'{runtime_dir}/results.npy', results_dict)

    print(f"\n结果已保存至: {runtime_dir}/results.npy")
    print("=" * 80)


# ====================== 其他函数保持不变 ======================
def train(args):
    """主训练函数 - 增强版（保持原样）"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.hop != 'auto':
        args.hop = int(args.hop)
    if args.max_nodes_per_hop is not None:
        args.max_nodes_per_hop = int(args.max_nodes_per_hop)

    # 准备数据
    args.file_dir = os.path.dirname(os.path.realpath('__file__'))

    if args.train_name is None:
        # 读取关联矩阵
        data = sio.loadmat(f'data/{args.data_name}-A.mat')
        net = data['net']

        # 读取相似度信息
        old_data = sio.loadmat(f'data/{args.data_name}dataset.mat')
        drug_sim = old_data["drug"].astype(float)
        disease_sim = old_data["disease"].astype(float)
        drug_num = drug_sim.shape[0]
        disease_num = disease_sim.shape[0]
        interactions = old_data["didr"].T
        drug_dis_matrix = np.mat(interactions)

        # 处理相似度矩阵
        row1, col1 = np.diag_indices_from(drug_sim)
        drug_sim[row1, col1] = 0
        row2, col2 = np.diag_indices_from(disease_sim)
        disease_sim[row2, col2] = 0

        dr_deg = drug_sim.sum(axis=1)
        dr_deg_inv_sqrt = dr_deg ** (-0.5)
        drug_sim = drug_sim * (dr_deg_inv_sqrt.reshape(-1, 1))
        drug_sim = drug_sim * (dr_deg_inv_sqrt.reshape(1, -1))
        drug_sim = drug_sim * args.simw

        dis_deg = disease_sim.sum(axis=1)
        dis_deg_inv_sqrt = dis_deg ** (-0.5)
        disease_sim = disease_sim * (dis_deg_inv_sqrt.reshape(-1, 1))
        disease_sim = disease_sim * (dis_deg_inv_sqrt.reshape(1, -1))
        disease_sim = disease_sim * args.simw

        # 构建特征矩阵
        mat1 = np.hstack((drug_sim, drug_dis_matrix))
        mat2 = np.hstack((drug_dis_matrix.T, disease_sim))
        attributes = np.vstack((mat1, mat2))

        # 读取异构网络
        data_heter = sio.loadmat(f'data/{args.data_name}-1.mat')
        net_heter = data_heter['net']

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
            metrics_result = SLGCN(args, fold_id, A, train_pos, train_neg,
                                   test_pos, test_neg, attributes)
            fold_results.append(metrics_result)

            gc.collect()

        # 输出结果
        print("\n" + "=" * 80)
        print("10-Fold Cross Validation Results - Enhanced SLGCN")
        print("=" * 80)

        all_metrics = np.array(fold_results)
        for i, metrics_result in enumerate(all_metrics):
            print(f"Fold {i + 1:2d}: Loss: {metrics_result[0]:.4f}, Acc: {metrics_result[1]:.4f}, "
                  f"AUC: {metrics_result[2]:.4f}, AUPR: {metrics_result[3]:.4f}")

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

def loop_dataset_batch(classifier, loader, g_list, optimizer=None, accumulation_steps=4):
    """批量训练循环 - 增强版"""
    total_loss = []
    all_targets = []
    all_scores = []
    all_acc = []

    classifier.train() if optimizer is not None else classifier.eval()

    batch_loss = 0
    step_count = 0

    for idx in loader:
        batch_scores = []
        batch_targets = []
        batch_acc = []

        for ind in idx:
            batch = g_list[ind]
            batch = batch.to(device)

            if optimizer is not None:  # 训练模式
                logits, loss, acc = classifier(batch)

                # 梯度累积
                loss = loss / accumulation_steps
                loss.backward()

                batch_loss += loss.item()
                batch_acc.append(acc)
            else:  # 评估模式
                with torch.no_grad():
                    logits, loss, acc = classifier(batch)
                    total_loss.append(loss.item())
                    batch_acc.append(acc)

            batch_scores.append(logits[:, 1].cpu().detach())
            batch_targets.extend(batch.y.tolist())

        # 优化器步骤（梯度累积）
        if optimizer is not None:
            step_count += 1
            if step_count % accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                # 记录累积的损失
                total_loss.append(batch_loss)
                batch_loss = 0

        all_scores.extend(batch_scores)
        all_targets.extend(batch_targets)
        all_acc.extend(batch_acc)

    # 处理剩余的梯度
    if optimizer is not None and step_count % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        total_loss.append(batch_loss)

    # 计算指标
    avg_loss = np.mean(total_loss) if len(total_loss) > 0 else 0
    avg_acc = np.mean(all_acc) if len(all_acc) > 0 else 0

    if len(all_scores) > 0:
        all_scores = torch.cat(all_scores).cpu().numpy()
    else:
        all_scores = np.array([])

    all_targets = np.array(all_targets)

    if len(all_targets) > 0 and len(all_scores) > 0:
        avg_precision = average_precision_score(all_targets, all_scores)
        fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
    else:
        avg_precision = 0
        auc = 0

    avg_metrics = np.array([float(avg_loss), float(avg_acc), float(auc), float(avg_precision)])
    return avg_metrics, all_scores
def get_optimizer_and_scheduler(model, args, total_steps):
    """获取优化器和学习率调度器"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    if args.use_cosine_scheduler:
        # 余弦退火调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.epoch // 4,
            T_mult=2,
            eta_min=args.lr * 0.01
        )
    else:
        # 带warmup的余弦退火
        def lr_lambda(current_step):
            if current_step < args.warmup_epochs * total_steps:
                return float(current_step) / float(max(1, args.warmup_epochs * total_steps))
            progress = float(current_step - args.warmup_epochs * total_steps) / \
                       float(max(1, (args.epoch - args.warmup_epochs) * total_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def SLGCN(args, fold, A, train_pos, train_neg, test_pos, test_neg, attributes=None):
    """训练和测试模型 - 简化版本"""
    print(f"\nFold {fold}: Starting subgraph extraction...")
    start_time = time.time()

    # 子图提取
    train_graphs, test_graphs, max_n_label = links2subgraphs(
        A, train_pos, train_neg, test_pos, test_neg, args.hop,
        args.max_nodes_per_hop, attributes, None, verbose=True
    )

    print(f"Fold {fold}: Subgraph extraction completed in {time.time() - start_time:.2f}s")

    # 转换为线图
    start_time = time.time()
    train_lines = to_linegraphs(train_graphs, max_n_label, verbose=True)
    test_lines = to_linegraphs(test_graphs, max_n_label, verbose=True)
    print(f"Fold {fold}: Line graph conversion completed in {time.time() - start_time:.2f}s")

    # 释放内存
    del train_graphs, test_graphs
    gc.collect()

    # 模型配置 - 使用简化版本
    hidden = args.hidden
    dropout = args.dropout_rate
    learning_rate = args.lr
    batch_size = args.batch_size
    feat_dim = (max_n_label + 1) * 2
    attr_dim = attributes.shape[1]
    latent_dim = [hidden // 2 for _ in range(args.layer)]

    # 确保latent_dim的维度合理
    latent_dim = [max(dim, 32) for dim in latent_dim]

    print(f"Fold {fold}: Simplified Model Configuration:")
    print(f"  Feature dimension: {feat_dim}")
    print(f"  Attribute dimension: {attr_dim}")
    print(f"  Hidden size: {hidden}")
    print(f"  Latent dimensions: {latent_dim}")
    print(f"  Training samples: {len(train_lines)}")
    print(f"  Test samples: {len(test_lines)}")

    # 检查一个样本的特征维度
    if len(train_lines) > 0:
        sample_data = train_lines[0]
        print(f"Sample data x shape: {sample_data.x.shape}")
        print(f"Total features: {sample_data.x.shape[1]}")
        print(f"Expected: feat_dim={feat_dim}, attr_dim={attr_dim}, total={feat_dim + attr_dim}")

    # 创建数据加载器
    train_idxes = torch.arange(len(train_lines)).long()
    test_idxes = torch.arange(len(test_lines)).long()

    train_loader = DataLoader(train_idxes, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_idxes, batch_size=1, shuffle=False)

    # 计算总步数
    total_steps = len(train_loader) * args.epoch

    # 选择模型 - 使用简化版本
    if args.model_type == 'enhanced':
        print(f"Fold {fold}: Using EnhancedSLGCN_Simple model")
        classifier = EnhancedSLGCN_Simple(
            feat_dim, hidden, attr_dim, latent_dim,
            dropout,
            use_layer_norm=args.use_layer_norm,
            use_residual=args.use_residual,
            label_smoothing=args.label_smoothing
        )
    elif args.model_type == 'hierarchical':
        print(f"Fold {fold}: Using HierarchicalSLGCN model")
        classifier = HierarchicalSLGCN(
            feat_dim, hidden, attr_dim, latent_dim,
            dropout,
            use_hierarchy=args.use_hierarchy
        )
    elif args.model_type == 'original_enhanced':
        print(f"Fold {fold}: Using EnhancedSLGCN_Original model")
        classifier = EnhancedSLGCN_Original(
            feat_dim, hidden, attr_dim, latent_dim,
            dropout,
            use_layer_norm=args.use_layer_norm,
            use_residual=args.use_residual,
            label_smoothing=args.label_smoothing
        )
    else:
        print(f"Fold {fold}: Using original SLGCN_model")
        classifier = SLGCN_model(feat_dim, hidden, attr_dim, latent_dim, dropout)

    classifier = classifier.to(device)

    # 测试前向传播
    print(f"\nTesting forward pass with a sample...")
    try:
        if len(train_lines) > 0:
            sample_batch = train_lines[0].to(device)
            with torch.no_grad():
                logits, loss, acc = classifier(sample_batch)
                print(f"Forward pass successful! Logits shape: {logits.shape}")
    except Exception as e:
        print(f"Error in forward pass: {e}")
        raise e

    # 优化器和调度器
    optimizer, scheduler = get_optimizer_and_scheduler(classifier, args, total_steps)

    # 训练循环
    best_auc = 0
    best_epoch = 0
    patience_counter = 0
    patience = args.patience

    # 记录训练历史
    train_history = []
    test_history = []

    for epoch in range(args.epoch):
        classifier.train()

        # 学习率预热
        if epoch < args.warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * (epoch + 1) / args.warmup_epochs

        # 训练
        train_metrics, _ = loop_dataset_batch(
            classifier, train_loader, train_lines, optimizer, args.grad_accumulation
        )

        # 验证
        classifier.eval()
        test_metrics, _ = loop_dataset_batch(
            classifier, test_loader, test_lines, None
        )

        # 更新学习率
        if epoch >= args.warmup_epochs:
            scheduler.step()

        # 记录历史
        train_history.append(train_metrics)
        test_history.append(test_metrics)

        # 打印进度
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Fold {fold}, Epoch {epoch + 1}/{args.epoch}: "
              f"LR: {current_lr:.6f}, "
              f"Train Loss: {train_metrics[0]:.4f}, Acc: {train_metrics[1]:.4f}, "
              f"Train AUC: {train_metrics[2]:.4f}, "
              f"Test Loss: {test_metrics[0]:.4f}, Acc: {test_metrics[1]:.4f}, "
              f"Test AUC: {test_metrics[2]:.4f}, AUPR: {test_metrics[3]:.4f}")

        # 保存最佳模型
        if test_metrics[2] > best_auc:
            best_auc = test_metrics[2]
            best_epoch = epoch
            patience_counter = 0
            torch.save(classifier.state_dict(),
                       f'runtime/model_fold_{fold}_best.pth')
            print(f"  ✓ New best model saved with AUC: {best_auc:.4f}")
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    # 加载最佳模型
    classifier.load_state_dict(
        torch.load(f'runtime/model_fold_{fold}_best.pth',
                   map_location=device)
    )
    classifier.eval()

    # 最终评估
    final_metrics, _ = loop_dataset_batch(
        classifier, test_loader, test_lines, None
    )

    print(f"\nFold {fold}: Final Results - "
          f"Loss: {final_metrics[0]:.4f}, Acc: {final_metrics[1]:.4f}, "
          f"AUC: {final_metrics[2]:.4f}, AUPR: {final_metrics[3]:.4f}")

    # 输出训练历史摘要
    train_history = np.array(train_history)
    test_history = np.array(test_history)

    print(f"  Best AUC: {best_auc:.4f} at epoch {best_epoch + 1}")
    print(f"  Average Train AUC: {train_history[:, 2].mean():.4f}")
    print(f"  Average Test AUC: {test_history[:, 2].mean():.4f}")

    # 清理内存
    del classifier, train_lines, test_lines
    gc.collect()

    return final_metrics

    return final_metrics
# ====================== 其他辅助函数保持不变 ======================
# 这里包含所有其他的函数定义，如 loop_dataset_batch, get_optimizer_and_scheduler, SLGCN 等
# 这些函数保持不变，只修改上面的部分

# 由于代码长度限制，这里省略其他辅助函数的完整代码，您原有的这些函数不需要修改
# 只需要确保上面定义的 train_with_integrated_data 函数正确即可

# ====================== 主程序入口 ======================
if __name__ == "__main__":
    args = parse(print_help=True)
    args.no_cuda = True
    args.cuda = False

    # 打印配置
    print("\n" + "=" * 70)
    print("增强版 SLGCN 模型 - 整合数据加载器")
    print("=" * 70)
    for key, value in vars(args).items():
        print(f"{key:25}: {value}")
    print("=" * 70 + "\n")

    # 选择训练函数
    if args.data_source == 'csv':
        print("使用CSV数据源进行训练...")
        train_with_integrated_data(args)
    else:
        print("使用MAT数据源进行训练...")
        train(args)  # 使用原来的训练函数