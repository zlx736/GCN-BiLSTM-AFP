import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv
from torch_geometric.data import DataLoader, Data
import pandas as pd
import numpy as np

# 模型定义
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

# 加载节点和边数据
df_nodes_test = pd.read_csv("Nodes_test.csv")
df_edges_test = pd.read_csv("Edges_test.csv")

unique_row_ids = df_nodes_test['row_id'].unique()
data_list = []
node_row_indices = []  # 用于追踪每个节点对应的原始 dataframe 的 index

for row_id in unique_row_ids:
    nodes = df_nodes_test[df_nodes_test['row_id'] == row_id]
    edges = df_edges_test[df_edges_test['row_id'] == row_id]

    # 选择正确的节点特征列（假设特征列从第3列到倒数第二列）
    x = torch.FloatTensor(nodes.iloc[:, 3:-1].values)  # 假设特征从第4列到倒数第2列
    source_nodes = edges['source_node'].to_numpy()
    target_nodes = edges['target_node'].to_numpy()
    edge_index = torch.tensor(np.array([source_nodes, target_nodes]), dtype=torch.long)

    # 选择正确的边特征列（假设边特征从第4列开始）
    edge_attr = torch.FloatTensor(edges.iloc[:, 3:].values)

    fake_y = torch.zeros(x.size(0), dtype=torch.long)  # 伪造标签，全为0

    # 这里保证了输入的维度是与模型预期的一致
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=fake_y)
    data_list.append(data)

    node_row_indices.extend(nodes.index.tolist())

# 创建 DataLoader
test_loader = DataLoader(data_list, batch_size=1, shuffle=False)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN_BiLSTMK(
    hidden_dim=16,
    hidden_dim2=32,
    lstm_layers=1,
    num_gcn_layers=2,
    num_node_features=26,
    num_edge_features=21
).to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 节点级预测
predicted_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        output = model(batch)  # [num_nodes, 2]
        preds = output.argmax(dim=1).cpu().numpy()
        predicted_labels.extend(preds)

# 添加预测列到原 CSV
df_nodes_test['label_pre'] = -1  # 初始化新列
df_nodes_test.loc[node_row_indices, 'label_pre'] = predicted_labels  # 写入预测值

# 保存结果
df_nodes_test.to_csv("Nodes_test_with_predictions.csv", index=False)
print("节点预测完成，结果已保存为 'Nodes_test_with_predictions.csv'")
# 读取原始的 CSV 文件
df = pd.read_csv("Nodes_test_with_predictions.csv")

# 只保留需要的列
df_filtered = df[['row_id', 'node_id', 'amino_acid', 'label', 'label_pre']]

# 保存为新的 CSV 文件
df_filtered.to_csv("Nodes_test_filtered.csv", index=False)

print("新的文件已保存为 'Nodes_test_filtered.csv'")
# 读取原始的 CSV 文件
df = pd.read_csv("Nodes_test_with_predictions.csv")

# 计算每个 row_id 中 label_pre 为 0 和 1 的数量，并为每个 row_id 生成唯一的 label_pre
def determine_label_pre(row_id):
    # 获取当前 row_id 对应的所有行
    subset = df[df['row_id'] == row_id]
    # 计算 label_pre 为 0 和 1 的数量
    count_0 = (subset['label_pre'] == 0).sum()
    count_1 = (subset['label_pre'] == 1).sum()
    
    # 根据 0 和 1 的数量确定最终的 label_pre
    if count_1 > count_0:
        return 1
    else:
        return 0

# 为每个 row_id 分配唯一的 label_pre
df['final_label_pre'] = df['row_id'].apply(determine_label_pre)

# 为每个 row_id 提取唯一的 label 和 label_pre
df_unique = df[['row_id', 'label', 'final_label_pre']].drop_duplicates()

# 统计 label 为 0 且 label_pre 为 1 的行数
label_0_and_pre_1 = ((df_unique['label'] == 0) & (df_unique['final_label_pre'] == 1)).sum()
label_0=(df_unique['label'] == 0).sum()
label_1=(df_unique['label'] == 1).sum()
print(f"label 为 0 的数量: {label_0}")
print(f"label 为 1 的数量: {label_1}")
# 统计 label 为 1 且 label_pre 为 0 的行数
label_1_and_pre_0 = ((df_unique['label'] == 1) & (df_unique['final_label_pre'] == 0)).sum()

# 输出结果
print(f"label 为 0 且 label_pre 为 1 的数量: {label_0_and_pre_1}")
print(f"label 为 1 且 label_pre 为 0 的数量: {label_1_and_pre_0}")

# 保存为新的 CSV 文件
df_unique.to_csv("Nodes_test_filtered_with_final_label_pre.csv", index=False)

print("新的文件已保存为 'Nodes_test_filtered_with_final_label_pre.csv'")