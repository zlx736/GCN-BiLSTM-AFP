import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv
from torch_geometric.data import DataLoader, Data
import pandas as pd
import numpy as np
from collections import Counter
import os
# 定义模型类
class GCN_BiLSTMK(nn.Module):
    def __init__(self, hidden_dim=64, hidden_dim2=30, lstm_layers=1, num_gcn_layers=2, num_node_features=26, num_edge_features=21):
        super(GCN_BiLSTMK, self).__init__()
        self.num_gcn_layers = num_gcn_layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_norms = nn.ModuleList()

        nn_first = nn.Sequential(
            nn.Linear(num_edge_features, num_node_features * hidden_dim)
        )
        self.gcn_layers.append(NNConv(num_node_features, hidden_dim, nn_first, aggr='mean'))
        self.gcn_norms.append(nn.LayerNorm(hidden_dim))

        for _ in range(1, num_gcn_layers):
            nn_other = nn.Sequential(
                nn.Linear(num_edge_features, hidden_dim * hidden_dim2)
            )
            self.gcn_layers.append(NNConv(hidden_dim, hidden_dim2, nn_other, aggr='mean'))
            self.gcn_norms.append(nn.LayerNorm(hidden_dim2))

        self.lstm = nn.LSTM(input_size=hidden_dim2, hidden_size=hidden_dim, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True)

        self.layer_norm_lstm = nn.LayerNorm(2 * hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2 * hidden_dim, 2)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for conv, norm in zip(self.gcn_layers, self.gcn_norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.leaky_relu(x)
            x = self.dropout(x)

        x = x.unsqueeze(0)
        x, _ = self.lstm(x)
        x = x.squeeze(0)
        x = self.layer_norm_lstm(x)
        x = self.fc(x)
        return x

    def soft_vote(self, node_preds, batch):
        """
    通过 Softmax 计算节点的类别概率，然后对每个图的节点概率进行加权，
    并根据加权后的平均概率来决定图级别的最终预测标签。
    """
    # 对每个节点的预测结果应用 Softmax 来得到概率分布
        node_probs = F.softmax(node_preds, dim=1)  # [num_nodes, num_classes]
    
    # 对每个图的节点进行加权平均计算，得到图的预测概率分布
        num_graphs = batch.max().item() + 1
        graph_probs = []

        for i in range(num_graphs):
        # 获取属于当前图的所有节点的概率
            node_probs_in_graph = node_probs[batch == i]
        
        # 获取每个节点的预测概率
            prob_0 = node_probs_in_graph[:, 0]  # 类别 0 的概率
            prob_1 = node_probs_in_graph[:, 1]  # 类别 1 的概率
        
        # 计算加权平均概率：给预测概率高的节点更多的权重
            weighted_prob_0 = (prob_0 * prob_0).sum()  # 对概率 0 的加权和
            weighted_prob_1 = (prob_1 * prob_1).sum()  # 对概率 1 的加权和
        
        # 计算图级别的最终概率：概率较高的类别会占主导地位
            if weighted_prob_1 > weighted_prob_0*2:
                graph_preds = 1
            else:
                graph_preds = 0
        
            graph_probs.append(graph_preds)
    
        return torch.tensor(graph_probs, device=node_preds.device, dtype=torch.long)
# 读取节点和边数据（无标签）
df_nodes_test = pd.read_csv("Bubalus_bubalis_processed_entries_100_N.csv")
df_edges_test = pd.read_csv("Bubalus_bubalis_processed_entries_100_E.csv")

# 创建一个列表来存储图数据
data_list = []
node_row_indices = []  # 用于追踪每个节点对应的原始 dataframe 的 index

# 按图（row_id）分割数据
for row_id in df_nodes_test['row_id'].unique():
    nodes_in_graph = df_nodes_test[df_nodes_test['row_id'] == row_id]
    edges_in_graph = df_edges_test[df_edges_test['row_id'] == row_id]

    x = torch.FloatTensor(nodes_in_graph.iloc[:, 3:].values)  
    source_nodes = edges_in_graph['source_node'].to_numpy()
    target_nodes = edges_in_graph['target_node'].to_numpy()
    edge_index = torch.tensor(np.array([source_nodes, target_nodes]), dtype=torch.long)

    edge_attr = torch.FloatTensor(edges_in_graph.iloc[:, 3:].values)

    fake_y = torch.zeros(x.size(0), dtype=torch.long)  # 伪造标签，全为0

    # 为每个图创建Data对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=fake_y)
    data_list.append(data)

    node_row_indices.extend(nodes_in_graph.index.tolist())

# 创建DataLoader
test_loader = DataLoader(data_list, batch_size=1, shuffle=False)
# 4. 加载训练好的模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN_BiLSTMK(
    hidden_dim=16,
    hidden_dim2=32,
    lstm_layers=1,
    num_gcn_layers=2,
    num_node_features=26,
    num_edge_features=21
).to(device)
model_path = 'best_model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

# 5. 使用模型对每个图进行预测
all_graph_preds = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        # 节点级别预测
        output = model(batch)
        # 图级别预测
        graph_pred = model.soft_vote(output, batch.batch)
        # 记录预测结果
        all_graph_preds.extend(graph_pred.cpu().tolist())

# 6. 读取 Apis_cerana_processed_entries_100.csv 原始数据
apis_cerana_df = pd.read_csv("Bubalus_bubalis_processed_entries_100.csv")


# 7. 添加预测的 label_pre 列到原始 DataFrame
apis_cerana_df['label_pre'] = all_graph_preds

# 8. 保存结果为新的 CSV 文件
output_file = "Bubalus_bubalis_processed_entries_100_with_label_pre.csv"
apis_cerana_df.to_csv(output_file, index=False)
print(f"带有 label_pre 的新文件已保存为 '{output_file}'")