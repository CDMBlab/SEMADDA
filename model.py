import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GINConv
import torch_geometric.nn as gnn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class Construction_Relation_Matrix(nn.Module):
    def __init__(self, act=nn.Sigmoid):
        super().__init__()
        self.act = act() if act is not None else nn.Identity()

    def forward(self, feature):
        x = feature @ feature.T
        return self.act(x)


class Structure_Enhanced_Message_Aggregation(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bi = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv_bi = 1. / math.sqrt(self.weight_bi.size(1))
        self.weight_bi.data.uniform_(-stdv_bi, stdv_bi)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv_bi, stdv_bi)

    def forward(self, input, adj, reconstruct_matrix):
        adj = torch.mul(adj, reconstruct_matrix)
        with torch.no_grad():
            row_sum = torch.sum(adj, dim=1, keepdim=True)
            r_inv = torch.pow(row_sum, -0.5).flatten()
            r_inv[torch.isinf(r_inv)] = 0.
            r_mat_inv = torch.diag(r_inv)

        adj = torch.matmul(r_mat_inv, adj)
        adj = torch.matmul(adj, r_mat_inv)

        support_bi = torch.mm(input, self.weight_bi)
        output = torch.spmm(adj, support_bi)

        if self.bias is not None:
            output = output + self.bias
        return output


class GraphAttentionLayer(nn.Module):
    """图注意力层 - 简化版本"""

    def __init__(self, in_features, out_features, dropout=0.4, alpha=0.2, concat=True):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)

        # 简化的注意力计算
        attention = torch.mm(Wh, Wh.t())
        attention = F.leaky_relu(attention, self.alpha)

        zero_vec = -9e15 * torch.ones_like(attention)
        attention = torch.where(adj > 0, attention, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class AdaptiveAttentionPooling(nn.Module):
    """自适应注意力池化层 - 自动调整维度"""

    def __init__(self, min_dim=32, max_dim=256):
        super().__init__()
        self.min_dim = min_dim
        self.max_dim = max_dim

        # 动态创建注意力层
        self.attention = None
        self.input_dim = None

    def build_attention(self, input_dim):
        """根据输入维度动态构建注意力层"""
        self.input_dim = input_dim
        hidden_dim = min(max(input_dim // 2, self.min_dim), self.max_dim)

        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        print(f"Built attention layer: input_dim={input_dim}, hidden_dim={hidden_dim}")

        # 初始化权重
        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, node_features):
        if node_features.size(0) == 0:
            return torch.zeros(node_features.size(1), device=node_features.device)

        # 如果还没有构建注意力层，或者输入维度变化了，重新构建
        if self.attention is None or self.input_dim != node_features.size(1):
            self.build_attention(node_features.size(1))

        attention_weights = F.softmax(self.attention(node_features), dim=0)
        graph_representation = torch.sum(attention_weights * node_features, dim=0)
        return graph_representation


class SimpleAttentionPooling(nn.Module):
    """简单的注意力池化层 - 使用点积注意力"""

    def __init__(self):
        super().__init__()
        # 不使用参数，使用简单的点积注意力

    def forward(self, node_features):
        if node_features.size(0) == 0:
            return torch.zeros(node_features.size(1), device=node_features.device)

        # 使用简单的点积注意力
        attention_scores = torch.matmul(node_features, node_features.t()).mean(dim=1, keepdim=True)
        attention_weights = F.softmax(attention_scores, dim=0)
        graph_representation = torch.sum(attention_weights * node_features, dim=0)
        return graph_representation


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失"""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        n_classes = logits.size(-1)
        log_preds = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            targets = torch.zeros_like(log_preds).scatter_(1, targets.unsqueeze(1), 1)
            targets = (1 - self.smoothing) * targets + self.smoothing / n_classes

        loss = -torch.sum(targets * log_preds, dim=-1).mean()
        return loss


class EnhancedSLGCN_Simple(nn.Module):
    """
    简化版SLGCN - 移除复杂的注意力池化，使用稳定架构
    """

    def __init__(self, input_dim, hidden_size, attr_dim, latent_dim=[64, 64, 64],
                 dropout=0.4, use_layer_norm=True, use_residual=True,
                 label_smoothing=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.attr_dim = attr_dim * 2
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.label_smoothing = label_smoothing

        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual

        print(f"Model config - input_dim: {input_dim}, hidden_size: {hidden_size}, "
              f"attr_dim: {attr_dim}, latent_dim: {latent_dim}")

        # 图卷积层 - 使用SAGEConv（稳定）
        self.conv_layers = nn.ModuleList()
        for i in range(len(latent_dim)):
            in_channels = input_dim if i == 0 else latent_dim[i - 1]
            out_channels = latent_dim[i]
            self.conv_layers.append(SAGEConv(in_channels, out_channels))

        # 层归一化
        if self.use_layer_norm:
            self.layer_norms = nn.ModuleList()
            for dim in latent_dim:
                self.layer_norms.append(nn.LayerNorm(dim))

        # 关系矩阵
        self.rm = Construction_Relation_Matrix(act=nn.Sigmoid)

        # 消息聚合层
        self.msg_layers = nn.ModuleList()
        for i in range(len(latent_dim)):
            in_dim = self.attr_dim if i == 0 else hidden_size
            self.msg_layers.append(
                Structure_Enhanced_Message_Aggregation(in_dim, hidden_size)
            )

        # 门控参数
        self.U_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.V_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        # 残差权重
        if self.use_residual:
            self.residual_weights = nn.ParameterList([
                nn.Parameter(torch.tensor(0.5)) for _ in range(len(latent_dim))
            ])

        # 简单的注意力池化（无参数）
        self.attention_pooling = SimpleAttentionPooling()

        # 投影层：将池化特征投影到hidden_size
        self.pooling_projection = nn.ModuleList()
        for dim in latent_dim:
            self.pooling_projection.append(nn.Linear(dim, hidden_size))

        # 分类器 - 简化版本
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(hidden_size // 2, 2)
        )

        # 相似度线性层
        self.linear_sim = nn.Linear(self.attr_dim, hidden_size)

        self._init_weights()
        self.dropout = nn.Dropout(dropout)

        # 损失函数
        if label_smoothing > 0:
            self.loss_fn = LabelSmoothingCrossEntropy(label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def _init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in [self.U_o, self.V_o, self.b_o]:
            weight.data.uniform_(-stdv, stdv)

        nn.init.xavier_uniform_(self.linear_sim.weight)
        nn.init.zeros_(self.linear_sim.bias)

        # 分类器初始化
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # 投影层初始化
        for proj in self.pooling_projection:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, data):
        x, edge_index, y = data.x, data.edge_index, data.y

        # 分离特征
        distance_feat = x[:, :self.input_dim]
        sim_feat = x[:, self.input_dim:self.input_dim + self.attr_dim]

        # 处理sim特征维度
        if sim_feat.shape[1] > self.attr_dim:
            sim_feat = sim_feat[:, :self.attr_dim]
        elif sim_feat.shape[1] < self.attr_dim:
            padding = torch.zeros(sim_feat.shape[0],
                                  self.attr_dim - sim_feat.shape[1],
                                  device=sim_feat.device)
            sim_feat = torch.cat([sim_feat, padding], dim=1)

        # 初始值
        cur_message = distance_feat
        output = sim_feat
        lin_sim = self.linear_sim(sim_feat)

        # 创建邻接矩阵（如果需要）
        num_nodes = x.shape[0]
        zero_adj = torch.zeros((num_nodes, num_nodes), device=x.device)

        if edge_index is not None and edge_index.shape[1] > 0:
            edge_index = edge_index.clone()
            edge_index[0] = torch.clamp(edge_index[0], 0, num_nodes - 1)
            edge_index[1] = torch.clamp(edge_index[1], 0, num_nodes - 1)

            zero_adj[edge_index[0], edge_index[1]] = 1
            zero_adj[edge_index[1], edge_index[0]] = 1

        # 归一化邻接矩阵
        with torch.no_grad():
            row_sum = torch.sum(zero_adj, dim=1, keepdim=True)
            r_inv = torch.pow(row_sum, -0.5).flatten()
            r_inv[torch.isinf(r_inv)] = 0.
            r_mat_inv = torch.diag(r_inv)

        adj_norm = torch.matmul(torch.matmul(r_mat_inv, zero_adj), r_mat_inv)

        # 保存所有层的特征
        all_layer_features = []

        # 多层处理
        for i in range(len(self.latent_dim)):
            # 保存残差
            residual_msg = cur_message.clone() if i > 0 and self.use_residual else None

            # 图卷积
            if edge_index is not None and edge_index.shape[1] > 0:
                cur_message = self.conv_layers[i](cur_message, edge_index)
            else:
                # 如果没有边，通过线性层变换维度
                if cur_message.shape[1] != self.latent_dim[i]:
                    linear_layer = nn.Linear(cur_message.shape[1], self.latent_dim[i]).to(cur_message.device)
                    cur_message = linear_layer(cur_message)

            # 层归一化
            if self.use_layer_norm:
                cur_message = self.layer_norms[i](cur_message)

            cur_message = F.relu(cur_message)

            # 残差连接
            if residual_msg is not None and self.use_residual:
                if residual_msg.shape[1] != cur_message.shape[1]:
                    # 如果维度不匹配，使用线性层投影
                    projection = nn.Linear(residual_msg.shape[1], cur_message.shape[1]).to(cur_message.device)
                    residual_msg = projection(residual_msg)

                alpha = torch.sigmoid(self.residual_weights[i])
                cur_message = alpha * cur_message + (1 - alpha) * residual_msg

            if i != len(self.latent_dim) - 1:
                cur_message = self.dropout(cur_message)

            # 保存当前层特征
            all_layer_features.append(cur_message)

            # 关系矩阵
            relation_matrix = self.rm(cur_message)

            # 消息聚合
            output = self.msg_layers[i](output, adj_norm, relation_matrix)

            # 门控机制
            o_t = torch.sigmoid(
                lin_sim @ self.U_o + output @ self.V_o + self.b_o
            )
            x_combined = lin_sim + output
            output = o_t * torch.tanh(x_combined)

            if i == len(self.latent_dim) - 1:
                output = F.relu(output)
            if i != len(self.latent_dim) - 1:
                output = self.dropout(output)

        # 注意力池化：使用最后一层特征
        if len(all_layer_features) > 0:
            last_layer_feat = all_layer_features[-1]

            # 池化
            pooled_feat = self.attention_pooling(last_layer_feat)

            # 投影到hidden_size
            final_pooled = self.pooling_projection[-1](pooled_feat.unsqueeze(0)).squeeze(0)
        else:
            final_pooled = torch.zeros(self.hidden_size, device=x.device)

        # 目标节点特征
        target_feature = output[0:1, :]

        # 结合池化特征和目标节点特征
        combined_feature = target_feature + final_pooled.unsqueeze(0)

        # 分类
        logits = self.classifier(combined_feature)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            y = Variable(y)
            loss = self.loss_fn(logits, y)
            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).float().mean().item()
            return logits, loss, acc

        return logits


class EnhancedSLGCN_Original(nn.Module):
    """
    原始SLGCN架构 + 简单改进（无复杂注意力）
    """

    def __init__(self, input_dim, hidden_size, attr_dim, latent_dim=[32, 32, 32],
                 dropout=0.4, use_layer_norm=True, use_residual=True,
                 label_smoothing=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.attr_dim = attr_dim * 2
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.label_smoothing = label_smoothing

        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual

        # 图卷积层
        self.conv_layers = nn.ModuleList()
        for i in range(len(latent_dim)):
            in_channels = input_dim if i == 0 else latent_dim[i - 1]
            self.conv_layers.append(SAGEConv(in_channels, latent_dim[i]))

        # 层归一化
        if self.use_layer_norm:
            self.layer_norms = nn.ModuleList()
            for dim in latent_dim:
                self.layer_norms.append(nn.LayerNorm(dim))

        # 关系矩阵
        self.rm = Construction_Relation_Matrix(act=nn.Sigmoid)

        # 消息聚合层
        self.msg_layers = nn.ModuleList()
        for i in range(len(latent_dim)):
            in_dim = self.attr_dim if i == 0 else hidden_size
            self.msg_layers.append(
                Structure_Enhanced_Message_Aggregation(in_dim, hidden_size)
            )

        # 门控参数
        self.U_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.V_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        # 残差权重
        if self.use_residual:
            self.residual_weights = nn.ParameterList([
                nn.Parameter(torch.tensor(0.5)) for _ in range(len(latent_dim))
            ])

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)
        )

        # 相似度线性层
        self.linear_sim = nn.Linear(self.attr_dim, hidden_size)

        self._init_weights()
        self.dropout = nn.Dropout(dropout)

        if label_smoothing > 0:
            self.loss_fn = LabelSmoothingCrossEntropy(label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def _init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in [self.U_o, self.V_o, self.b_o]:
            weight.data.uniform_(-stdv, stdv)

        nn.init.xavier_uniform_(self.linear_sim.weight)
        nn.init.zeros_(self.linear_sim.bias)

        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, data):
        x, edge_index, y = data.x, data.edge_index, data.y

        # 分离特征
        distance_feat = x[:, :self.input_dim]
        sim_feat = x[:, self.input_dim:self.input_dim + self.attr_dim]

        # 处理sim特征维度
        if sim_feat.shape[1] > self.attr_dim:
            sim_feat = sim_feat[:, :self.attr_dim]
        elif sim_feat.shape[1] < self.attr_dim:
            padding = torch.zeros(sim_feat.shape[0],
                                  self.attr_dim - sim_feat.shape[1],
                                  device=sim_feat.device)
            sim_feat = torch.cat([sim_feat, padding], dim=1)

        # 初始值
        cur_message = distance_feat
        output = sim_feat
        lin_sim = self.linear_sim(sim_feat)

        # 创建邻接矩阵
        num_nodes = x.shape[0]
        zero_adj = torch.zeros((num_nodes, num_nodes), device=x.device)
        if edge_index is not None and edge_index.shape[1] > 0:
            zero_adj[edge_index[0], edge_index[1]] = 1
            zero_adj[edge_index[1], edge_index[0]] = 1

        # 归一化邻接矩阵
        with torch.no_grad():
            row_sum = torch.sum(zero_adj, dim=1, keepdim=True)
            r_inv = torch.pow(row_sum, -0.5).flatten()
            r_inv[torch.isinf(r_inv)] = 0.
            r_mat_inv = torch.diag(r_inv)

        adj_norm = torch.matmul(torch.matmul(r_mat_inv, zero_adj), r_mat_inv)

        # 多层处理
        for i in range(len(self.latent_dim)):
            # 保存残差
            residual_msg = cur_message.clone() if i > 0 and self.use_residual else None

            # 图卷积
            if edge_index is not None and edge_index.shape[1] > 0:
                cur_message = self.conv_layers[i](cur_message, edge_index)
            else:
                cur_message = cur_message

            # 层归一化
            if self.use_layer_norm:
                cur_message = self.layer_norms[i](cur_message)

            cur_message = F.relu(cur_message)

            # 残差连接
            if residual_msg is not None:
                alpha = torch.sigmoid(self.residual_weights[i])
                cur_message = alpha * cur_message + (1 - alpha) * residual_msg

            if i != len(self.latent_dim) - 1:
                cur_message = self.dropout(cur_message)

            # 关系矩阵
            relation_matrix = self.rm(cur_message)

            # 消息聚合
            output = self.msg_layers[i](output, adj_norm, relation_matrix)

            # 门控机制
            o_t = torch.sigmoid(
                lin_sim @ self.U_o + output @ self.V_o + self.b_o
            )
            x_combined = lin_sim + output
            output = o_t * torch.tanh(x_combined)

            if i == len(self.latent_dim) - 1:
                output = F.relu(output)
            if i != len(self.latent_dim) - 1:
                output = self.dropout(output)

        # 分类
        target_feature = output[0:1, :]
        logits = self.classifier(target_feature)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            y = Variable(y)
            loss = self.loss_fn(logits, y)
            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).float().mean().item()
            return logits, loss, acc

        return logits


class HierarchicalSLGCN(nn.Module):
    """层次化SLGCN（保持原版）"""

    def __init__(self, input_dim, hidden_size, attr_dim, latent_dim=[32, 32, 32],
                 dropout=0.4, use_hierarchy=True):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.attr_dim = attr_dim * 2
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.use_hierarchy = use_hierarchy

        # 图卷积层
        self.conv_layers = nn.ModuleList()
        for i in range(len(latent_dim)):
            in_channels = input_dim if i == 0 else latent_dim[i - 1]
            self.conv_layers.append(SAGEConv(in_channels, latent_dim[i]))

        # 关系矩阵
        self.rm = Construction_Relation_Matrix(act=nn.Sigmoid)

        # 消息聚合层
        self.msg_layers = nn.ModuleList()
        for i in range(len(latent_dim)):
            in_dim = self.attr_dim if i == 0 else hidden_size
            self.msg_layers.append(
                Structure_Enhanced_Message_Aggregation(in_dim, hidden_size)
            )

        # 门控参数
        self.U_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.V_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)
        )

        # 相似度线性层
        self.linear_sim = nn.Linear(self.attr_dim, hidden_size)

        self._init_weights()
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in [self.U_o, self.V_o, self.b_o]:
            weight.data.uniform_(-stdv, stdv)

        nn.init.xavier_uniform_(self.linear_sim.weight)
        nn.init.zeros_(self.linear_sim.bias)
        nn.init.xavier_uniform_(self.classifier[0].weight)
        nn.init.zeros_(self.classifier[0].bias)
        nn.init.xavier_uniform_(self.classifier[3].weight)
        nn.init.zeros_(self.classifier[3].bias)

    def forward(self, data):
        x, edge_index, y = data.x, data.edge_index, data.y

        # 分离特征
        distance_feat = x[:, :self.input_dim]
        sim_feat = x[:, self.input_dim:self.input_dim + self.attr_dim]

        # 处理sim特征维度
        if sim_feat.shape[1] > self.attr_dim:
            sim_feat = sim_feat[:, :self.attr_dim]
        elif sim_feat.shape[1] < self.attr_dim:
            padding = torch.zeros(sim_feat.shape[0],
                                  self.attr_dim - sim_feat.shape[1],
                                  device=sim_feat.device)
            sim_feat = torch.cat([sim_feat, padding], dim=1)

        # 初始值
        cur_message = distance_feat
        output = sim_feat
        lin_sim = self.linear_sim(sim_feat)

        # 创建邻接矩阵
        num_nodes = x.shape[0]
        zero_adj = torch.zeros((num_nodes, num_nodes), device=x.device)
        if edge_index is not None and edge_index.shape[1] > 0:
            zero_adj[edge_index[0], edge_index[1]] = 1
            zero_adj[edge_index[1], edge_index[0]] = 1

        # 归一化邻接矩阵
        with torch.no_grad():
            row_sum = torch.sum(zero_adj, dim=1, keepdim=True)
            r_inv = torch.pow(row_sum, -0.5).flatten()
            r_inv[torch.isinf(r_inv)] = 0.
            r_mat_inv = torch.diag(r_inv)

        adj_norm = torch.matmul(torch.matmul(r_mat_inv, zero_adj), r_mat_inv)

        # 多层处理
        for i in range(len(self.latent_dim)):
            # 图卷积
            if edge_index is not None and edge_index.shape[1] > 0:
                cur_message = self.conv_layers[i](cur_message, edge_index)
            else:
                cur_message = cur_message

            if i != len(self.latent_dim) - 1:
                cur_message = self.dropout(cur_message)

            # 关系矩阵
            relation_matrix = self.rm(cur_message)

            # 消息聚合
            output = self.msg_layers[i](output, adj_norm, relation_matrix)

            # 门控机制
            o_t = torch.sigmoid(
                lin_sim @ self.U_o + output @ self.V_o + self.b_o
            )
            x_combined = lin_sim + output
            output = o_t * torch.tanh(x_combined)

            if i == len(self.latent_dim) - 1:
                output = F.relu(output)
            if i != len(self.latent_dim) - 1:
                output = self.dropout(output)

        # 分类
        target_feature = output[0:1, :]
        logits = self.classifier(target_feature)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)
            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).float().mean().item()
            return logits, loss, acc

        return logits


class SLGCN_model(nn.Module):
    """原始SLGCN模型（保持原样）"""

    def __init__(self, input_dim, hidden_size, attr_dim, latent_dim=[32, 32, 32], dropout=0.4):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.attr_dim = attr_dim * 2
        self.hidden_size = hidden_size

        # 图卷积层
        self.conv_layers = nn.ModuleList()
        for i in range(len(latent_dim)):
            in_channels = input_dim if i == 0 else latent_dim[i - 1]
            self.conv_layers.append(SAGEConv(in_channels, latent_dim[i]))

        # 关系矩阵
        self.rm = Construction_Relation_Matrix(act=nn.Sigmoid)

        # 消息聚合层
        self.msg_layers = nn.ModuleList()
        for i in range(len(latent_dim)):
            in_dim = self.attr_dim if i == 0 else hidden_size
            self.msg_layers.append(
                Structure_Enhanced_Message_Aggregation(in_dim, hidden_size)
            )

        # 门控参数
        self.U_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.V_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        # 分类器
        self.classifier = nn.Linear(hidden_size, 2)

        # 相似度线性层
        self.linear_sim = nn.Linear(self.attr_dim, hidden_size)

        self._init_weights()
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in [self.U_o, self.V_o, self.b_o]:
            weight.data.uniform_(-stdv, stdv)

        nn.init.xavier_uniform_(self.linear_sim.weight)
        nn.init.zeros_(self.linear_sim.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, data):
        x, edge_index, y = data.x, data.edge_index, data.y

        # 分离特征
        distance_feat = x[:, :self.input_dim]
        sim_feat = x[:, self.input_dim:self.input_dim + self.attr_dim]

        # 处理sim特征维度
        if sim_feat.shape[1] > self.attr_dim:
            sim_feat = sim_feat[:, :self.attr_dim]
        elif sim_feat.shape[1] < self.attr_dim:
            padding = torch.zeros(sim_feat.shape[0],
                                  self.attr_dim - sim_feat.shape[1],
                                  device=sim_feat.device)
            sim_feat = torch.cat([sim_feat, padding], dim=1)

        # 初始值
        cur_message = distance_feat
        output = sim_feat
        lin_sim = self.linear_sim(sim_feat)

        # 创建邻接矩阵
        num_nodes = x.shape[0]
        zero_adj = torch.zeros((num_nodes, num_nodes), device=x.device)
        if edge_index is not None and edge_index.shape[1] > 0:
            zero_adj[edge_index[0], edge_index[1]] = 1
            zero_adj[edge_index[1], edge_index[0]] = 1

        # 归一化邻接矩阵
        with torch.no_grad():
            row_sum = torch.sum(zero_adj, dim=1, keepdim=True)
            r_inv = torch.pow(row_sum, -0.5).flatten()
            r_inv[torch.isinf(r_inv)] = 0.
            r_mat_inv = torch.diag(r_inv)

        adj_norm = torch.matmul(torch.matmul(r_mat_inv, zero_adj), r_mat_inv)

        # 多层处理
        for i in range(len(self.latent_dim)):
            # 图卷积
            if edge_index is not None and edge_index.shape[1] > 0:
                cur_message = self.conv_layers[i](cur_message, edge_index)
            else:
                cur_message = cur_message

            if i != len(self.latent_dim) - 1:
                cur_message = self.dropout(cur_message)

            # 关系矩阵
            relation_matrix = self.rm(cur_message)

            # 消息聚合
            output = self.msg_layers[i](output, adj_norm, relation_matrix)

            # 门控机制
            o_t = torch.sigmoid(
                lin_sim @ self.U_o + output @ self.V_o + self.b_o
            )
            x_combined = lin_sim + output
            output = o_t * torch.tanh(x_combined)

            if i == len(self.latent_dim) - 1:
                output = F.relu(output)
            if i != len(self.latent_dim) - 1:
                output = self.dropout(output)

        # 分类
        target_feature = output[0:1, :]
        logits = self.classifier(target_feature)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)
            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).float().mean().item()
            return logits, loss, acc

        return logits


