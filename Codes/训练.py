import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import NNConv
from torch_geometric.data import DataLoader, Data
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score,precision_score
from collections import Counter

# 加载节点和边数据
df_nodes = pd.read_csv("Nodes1.csv")  # 加载节点特征
df_edges = pd.read_csv("Edges1.csv")  # 加载边特征

# 创建节点 ID 和特征的映射
node_id_to_index = {node_id: idx for idx, node_id in enumerate(df_nodes['node_id'])}

# 获取所有唯一的 row_id，即每张图的标识符
unique_row_ids = df_nodes['row_id'].unique()

# 用来存储每张图的 Data 对象
data_list = []

for row_id in unique_row_ids:
    # 筛选出属于当前图的节点和边
    nodes_in_graph = df_nodes[df_nodes['row_id'] == row_id]
    edges_in_graph = df_edges[df_edges['row_id'] == row_id]

    # 获取节点特征
    node_features = nodes_in_graph.iloc[:, 3:-1].values  # 假设从第4列到倒数第二列是特征
    x = torch.FloatTensor(node_features)

    # 获取 'label' 列作为目标值
    y = torch.LongTensor(nodes_in_graph["label"].values)

    # 获取边的索引（确保源节点和目标节点在当前图的节点范围内）
    source_nodes = edges_in_graph['source_node'].to_numpy()
    target_nodes = edges_in_graph['target_node'].to_numpy()

    # 生成边索引
    edge_index = torch.tensor(np.array([source_nodes, target_nodes]), dtype=torch.long)

    # 生成边特征（假设边特征从第4列开始，去掉 'row_id'、'source_node' 和 'target_node' 列）
    edge_features = edges_in_graph.iloc[:, 3:].values
    edge_attr = torch.FloatTensor(edge_features)

    # 创建 PyG Data 对象
    data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
    data_list.append(data)

# 图数据填充：确保每个图的节点数相同
def pad_graphs(data_list, max_nodes):
    padded_data_list = []
    
    for data in data_list:
        num_nodes = data.x.size(0)
        
        # 填充节点特征 x
        if num_nodes < max_nodes:
            padding = torch.zeros(max_nodes - num_nodes, data.x.size(1))  # 生成填充
            padded_x = torch.cat([data.x, padding], dim=0)
        else:
            padded_x = data.x

        # 填充节点标签 y
        if num_nodes < max_nodes:
            padding = torch.full((max_nodes - num_nodes,), -1, dtype=torch.long)  # 填充占位符 -1
            padded_y = torch.cat([data.y, padding], dim=0)
        else:
            padded_y = data.y
        # 填充边索引 edge_index
        edge_index = data.edge_index
        if num_nodes < max_nodes:
            # 更新边索引，确保它们指向填充后的节点
            edge_index = edge_index.clone()
            edge_index[edge_index >= num_nodes] += (max_nodes - num_nodes)  # 调整超出范围的索引
        else:
            edge_index = data.edge_index
        # 创建新的 Data 对象并加入列表
        padded_data = Data(x=padded_x, edge_index=edge_index, edge_attr=data.edge_attr, y=padded_y)
        padded_data_list.append(padded_data)
    return padded_data_list

# 找到所有图中最大节点数
max_nodes = max([data.x.size(0) for data in data_list])  # 计算最大节点数

# 使用填充处理图数据
padded_data_list = pad_graphs(data_list, max_nodes)

# 数据集划分（80% 训练集，20% 验证集）
train_data_list, val_data_list = train_test_split(padded_data_list, test_size=0.2, random_state=42)
print(len(train_data_list))
print(len(val_data_list))

# 创建 DataLoader
train_loader = DataLoader(train_data_list, batch_size=100, shuffle=True)
val_loader = DataLoader(val_data_list, batch_size=10, shuffle=True)

# GCN + Bi-LSTM 模型定义
class GCN_BiLSTMK(nn.Module):
    def __init__(self, hidden_dim=64, hidden_dim2=30, lstm_layers=1, num_gcn_layers=2, num_node_features=26, num_edge_features=21):
        super(GCN_BiLSTMK, self).__init__()
        self.num_gcn_layers = num_gcn_layers

        self.gcn_layers = nn.ModuleList()
        self.gcn_norms = nn.ModuleList()

        # 第一个 GCN 层
        nn_first = nn.Sequential(
            nn.Linear(num_edge_features, num_node_features * hidden_dim)
        )
        self.gcn_layers.append(NNConv(num_node_features, hidden_dim, nn_first, aggr='mean'))
        self.gcn_norms.append(nn.LayerNorm(hidden_dim))  # 对应第一层输出维度

        # 后续 GCN 层
        for _ in range(1, num_gcn_layers):
            nn_other = nn.Sequential(
                nn.Linear(num_edge_features, hidden_dim * hidden_dim2)
            )
            self.gcn_layers.append(NNConv(hidden_dim, hidden_dim2, nn_other, aggr='mean'))
            self.gcn_norms.append(nn.LayerNorm(hidden_dim2))  # 对应后续层输出维度

        # Bi-LSTM
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

        x = x.unsqueeze(0)  # [1, num_nodes, hidden_dim2]
        x, _ = self.lstm(x)
        x = x.squeeze(0)  # [num_nodes, 2*hidden_dim]
        x = self.layer_norm_lstm(x)
        x = self.fc(x)  # [num_nodes, 2]

        return x

    def majority_vote(self, node_preds, batch):
        """
        对每个图内的所有节点预测进行少数服从多数的操作，
        使得每个图的所有节点的预测标签一致。
        """
        pred_labels = node_preds.argmax(dim=1)  # [total_num_nodes]
        num_graphs = batch.max().item() + 1     # 图的数量
        graph_preds = []

        for i in range(num_graphs):
            node_labels = pred_labels[batch == i]  # 当前图的所有节点的预测标签
            most_common = Counter(node_labels.tolist()).most_common(1)[0][0]
            graph_preds.append(most_common)

        return torch.tensor(graph_preds, device=node_preds.device, dtype=torch.long)



def train_and_evaluate(train_loader, val_loader, model, optimizer, criterion, scheduler, device, num_epochs=200):
    best_val_loss = float('inf')
    best_metrics = None
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        total_train_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # 计算模型输出
            output = model(batch)

            # 创建一个mask，忽略y=-1的节点
            mask = batch.y != -1  # True for valid nodes, False for padded nodes

            # 计算损失：只计算有效节点
            loss = criterion(output[mask], batch.y[mask])
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # === 验证阶段 ===
        model.eval()  # 设置模型为评估模式
        total_val_loss = 0
        all_preds = []
        all_labels = []
        valid_nodes_count = 0  # 计算有效节点数

        with torch.no_grad():  # 不计算梯度
            for idx, batch in enumerate(val_loader):
                batch = batch.to(device)

                # 计算模型输出
                output = model(batch)

                # 创建一个mask，忽略y=-1的节点
                mask = batch.y != -1  # True for valid nodes, False for padded nodes

                # 计算有效节点的损失
                val_loss = criterion(output[mask], batch.y[mask])
                total_val_loss += val_loss.item()

                # 对有效节点进行预测和标签收集
                preds = output.argmax(dim=1).cpu().numpy()
                labels = batch.y.cpu().numpy()

                # 只收集有效节点的预测和标签
                valid_preds = preds[mask.cpu()]
                valid_labels = labels[mask.cpu()]

                all_preds.extend(valid_preds)
                all_labels.extend(valid_labels)

                # 累加有效节点的数量
                valid_nodes_count += mask.sum().item()

        avg_val_loss = total_val_loss / len(val_loader)

        # 计算评估指标时去除图内真实节点数量带来的影响
        accuracy = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')  # 计算Precision
        try:
            auc = roc_auc_score(all_labels, all_preds, multi_class='ovo', average='macro')
        except ValueError:
            auc = 0.0

        # 计算加权指标
        weighted_accuracy = accuracy * (valid_nodes_count / len(all_labels))
        weighted_recall = recall * (valid_nodes_count / len(all_labels))
        weighted_f1 = f1 * (valid_nodes_count / len(all_labels))
        weighted_precision = precision * (valid_nodes_count / len(all_labels))  # 加权Precision
        weighted_auc = auc * (valid_nodes_count / len(all_labels))

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Validation Metrics - Accuracy: {weighted_accuracy:.4f}, Recall: {weighted_recall:.4f}, F1: {weighted_f1:.4f}, Precision: {weighted_precision:.4f}, AUC: {weighted_auc:.4f}")

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_metrics = (weighted_accuracy, weighted_recall, weighted_f1, weighted_precision, weighted_auc)
            torch.save(model.state_dict(), 'best_model-1632.pth')
            print("▲[Model saved]▲")

        if (epoch + 1) % 50 == 0 and best_metrics is not None:
            acc, rec, f1s, prec, aucv = best_metrics
            print(f"[Best Model] Epoch {best_epoch}: Accuracy: {acc:.4f}, Recall: {rec:.4f}, F1: {f1s:.4f}, Precision: {prec:.4f}, AUC: {aucv:.4f}")

        scheduler.step()  # 更新学习率

# 初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN_BiLSTMK(
    hidden_dim=16,
    hidden_dim2=32,
    #hidden_dim=8,
    #hidden_dim2=16,
    lstm_layers=1,
    num_gcn_layers=2,
    num_node_features=26,  # 节点特征数
    num_edge_features=21   # 边特征数
).to(device)

# 损失函数 & 优化器 & 学习率调度器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=5e-4)

def lr_lambda(epoch):
    # 学习率下降的公式，确保学习率不会小于 5e-4
    min_lr = 5e-4
    initial_lr = 1e-2
    new_lr = initial_lr * (0.7 ** (epoch // 50))  # 每30个epoch下降一次
    return max(new_lr / initial_lr, min_lr / initial_lr)  # 保证学习率不会低于 5e-4

# 使用 LambdaLR 调度器
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# 训练
train_and_evaluate(train_loader, val_loader, model, optimizer, criterion, scheduler, device, num_epochs=300)