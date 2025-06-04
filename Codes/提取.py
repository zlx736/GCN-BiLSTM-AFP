from Bio import SeqIO
import pandas as pd

# 读取FASTA文件并提取序列和长度
fasta_file = "Bubalus_bubalis_processed_entries_100.fasta"

sequences = []
for record in SeqIO.parse(fasta_file, "fasta"):
    sequence = str(record.seq)
    length = len(sequence)
    sequences.append({"sequence": sequence, "length": length})

# 创建DataFrame并保存为CSV文件
df = pd.DataFrame(sequences)
csv_file_path = "Bubalus_bubalis_processed_entries_100.csv"
df.to_csv(csv_file_path, index=False)

csv_file_path
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os# 20 种常见氨基酸
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

# 创建 aa_to_index 字典（氨基酸到索引的映射）
aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}

# 创建 One-Hot 编码矩阵
one_hot = np.eye(len(amino_acids))
# 手动定义 BLOSUM62 矩阵（只包含 20 种常见氨基酸）
blosum62 = {
    ('A', 'A'): 4,  ('A', 'R'): -1, ('A', 'N'): -2, ('A', 'D'): -2, ('A', 'C'): 0,
    ('A', 'Q'): -1, ('A', 'E'): -1, ('A', 'G'): 0,  ('A', 'H'): -2, ('A', 'I'): -1,
    ('A', 'L'): -1, ('A', 'K'): -1, ('A', 'M'): -1, ('A', 'F'): -2, ('A', 'P'): -1,
    ('A', 'S'): 1,  ('A', 'T'): 0,  ('A', 'W'): -3, ('A', 'Y'): -2, ('A', 'V'): 0,

    ('R', 'R'): 5,  ('R', 'N'): 0,  ('R', 'D'): -2, ('R', 'C'): -3, ('R', 'Q'): 1,
    ('R', 'E'): 0,  ('R', 'G'): -2, ('R', 'H'): 0,  ('R', 'I'): -2, ('R', 'L'): -2,
    ('R', 'K'): 2,  ('R', 'M'): -1, ('R', 'F'): -3, ('R', 'P'): -1, ('R', 'S'): 0,
    ('R', 'T'): -1, ('R', 'W'): -3, ('R', 'Y'): -2, ('R', 'V'): -2,

    ('N', 'N'): 6,  ('N', 'D'): 1,  ('N', 'C'): -3, ('N', 'Q'): 0,  ('N', 'E'): 0,
    ('N', 'G'): 0,  ('N', 'H'): 1,  ('N', 'I'): -3, ('N', 'L'): -3, ('N', 'K'): 0,
    ('N', 'M'): -2, ('N', 'F'): -3, ('N', 'P'): -2, ('N', 'S'): 1,  ('N', 'T'): 0,
    ('N', 'W'): -4, ('N', 'Y'): -2, ('N', 'V'): -3,

    ('D', 'D'): 6,  ('D', 'C'): -4, ('D', 'Q'): 1,  ('D', 'E'): 2,  ('D', 'G'): 0,
    ('D', 'H'): -1, ('D', 'I'): -3, ('D', 'L'): -4, ('D', 'K'): 0,  ('D', 'M'): -3,
    ('D', 'F'): -4, ('D', 'P'): -1, ('D', 'S'): 0,  ('D', 'T'): -1, ('D', 'W'): -4,
    ('D', 'Y'): -3, ('D', 'V'): -3,

    ('C', 'C'): 9,  ('C', 'Q'): -3, ('C', 'E'): -4, ('C', 'G'): -3, ('C', 'H'): -3,
    ('C', 'I'): -1, ('C', 'L'): -1, ('C', 'K'): -3, ('C', 'M'): -1, ('C', 'F'): -2,
    ('C', 'P'): -3, ('C', 'S'): -1, ('C', 'T'): -1, ('C', 'W'): -2, ('C', 'Y'): -2,
    ('C', 'V'): -1,

    ('Q', 'Q'): 5,  ('Q', 'E'): 2,  ('Q', 'G'): -2, ('Q', 'H'): 0,  ('Q', 'I'): -3,
    ('Q', 'L'): -2, ('Q', 'K'): 1,  ('Q', 'M'): 0,  ('Q', 'F'): -3, ('Q', 'P'): -1,
    ('Q', 'S'): 0,  ('Q', 'T'): -1, ('Q', 'W'): -2, ('Q', 'Y'): -1, ('Q', 'V'): -2,

    ('E', 'E'): 5,  ('E', 'G'): -2, ('E', 'H'): -1, ('E', 'I'): -3, ('E', 'L'): -3,
    ('E', 'K'): 1,  ('E', 'M'): -2, ('E', 'F'): -3, ('E', 'P'): -1, ('E', 'S'): 0,
    ('E', 'T'): -1, ('E', 'W'): -3, ('E', 'Y'): -2, ('E', 'V'): -2,

    ('G', 'G'): 6,  ('G', 'H'): -2, ('G', 'I'): -4, ('G', 'L'): -4, ('G', 'K'): -2,
    ('G', 'M'): -3, ('G', 'F'): -3, ('G', 'P'): -2, ('G', 'S'): 0,  ('G', 'T'): -2,
    ('G', 'W'): -2, ('G', 'Y'): -3, ('G', 'V'): -3,

    ('H', 'H'): 8,  ('H', 'I'): -3, ('H', 'L'): -3, ('H', 'K'): -1, ('H', 'M'): -2,
    ('H', 'F'): -1, ('H', 'P'): -2, ('H', 'S'): -1, ('H', 'T'): -2, ('H', 'W'): 2,
    ('H', 'Y'): 0,  ('H', 'V'): -3,

    ('I', 'I'): 4,  ('I', 'L'): 2,  ('I', 'K'): -2, ('I', 'M'): 1,  ('I', 'F'): 0,
    ('I', 'P'): -3, ('I', 'S'): -2, ('I', 'T'): -1, ('I', 'W'): -3, ('I', 'Y'): -1,
    ('I', 'V'): 3,

    ('L', 'L'): 4,  ('L', 'K'): -2, ('L', 'M'): 2,  ('L', 'F'): 0,  ('L', 'P'): -3,
    ('L', 'S'): -2, ('L', 'T'): -2, ('L', 'W'): -2, ('L', 'Y'): -1, ('L', 'V'): 1,

    ('K', 'K'): 5,  ('K', 'M'): -1, ('K', 'F'): -3, ('K', 'P'): -1, ('K', 'S'): 0,
    ('K', 'T'): -1, ('K', 'W'): -3, ('K', 'Y'): -2, ('K', 'V'): -2,

    ('M', 'M'): 5,  ('M', 'F'): -1, ('M', 'P'): -2, ('M', 'S'): -1, ('M', 'T'): -1,
    ('M', 'W'): -1, ('M', 'Y'): -1, ('M', 'V'): 1,

    ('F', 'F'): 6,  ('F', 'P'): -3, ('F', 'S'): -3, ('F', 'T'): -2, ('F', 'W'): 1,
    ('F', 'Y'): 3,  ('F', 'V'): -1,

    ('P', 'P'): 7,  ('P', 'S'): -1, ('P', 'T'): -1, ('P', 'W'): -4, ('P', 'Y'): -3,
    ('P', 'V'): -2,

    ('S', 'S'): 4,  ('S', 'T'): 1,  ('S', 'W'): -3, ('S', 'Y'): -2, ('S', 'V'): -2,

    ('T', 'T'): 5,  ('T', 'W'): -2, ('T', 'Y'): -2, ('T', 'V'): -1,

    ('W', 'W'): 11, ('W', 'Y'): 2,  ('W', 'V'): -3,

    ('Y', 'Y'): 7,  ('Y', 'V'): -1,

    ('V', 'V'): 4
}
# 自动补充对称矩阵
for (a1, a2), score in list(blosum62.items()):
    if (a2, a1) not in blosum62:
        blosum62[(a2, a1)] = score
# 获取BLOSUM62分数
def get_blosum62_score(a1, a2):
    return blosum62.get((a1, a2), 0)  # 如果没有该替代对，则返回 0
# 完整的Z-scale特征（Sandberg et al., 1998）
zscale_data = {
    'A': [0.24, -2.32, 0.60, -0.14, 1.30],
    'R': [3.52, 2.50, -3.50, 1.99, -0.17],
    'N': [3.05, 1.62, 1.04, -1.15, 1.61],
    'D': [3.98, 0.93, 1.93, -2.46, 0.75],
    'C': [0.84, -1.67, 3.71, 0.18, -2.65],
    'Q': [1.75, 0.50, -1.44, -1.34, 0.66],
    'E': [3.11, 0.26, -0.11, -3.04, -0.25],
    'G': [2.05, -4.06, 0.36, -0.82, -0.38],
    'H': [2.47, 1.95, 0.26, 3.90, 0.09],
    'I': [-3.89, -1.73, -1.71, -0.84, 0.26],
    'L': [-4.28, -1.30, -1.49, -0.72, 0.84],
    'K': [2.29, 0.89, -2.49, 1.49, 0.31],
    'M': [-2.85, -0.22, 0.47, 1.94, -0.98],
    'F': [-4.22, 1.94, 1.06, 0.54, -0.62],
    'P': [-1.66, 0.27, 1.84, 0.70, 2.00],
    'S': [2.39, -1.07, 1.15, -1.39, 0.67],
    'T': [0.75, -2.18, -1.12, -1.46, -0.40],
    'W': [-4.36, 3.94, 0.59, 3.44, -1.59],
    'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],
    'V': [-2.59, -2.65, -1.29, -0.28, -0.33],
}
def compute_ngram_probs(seq, k=1):
    """
    计算给定氨基酸序列中每个氨基酸后接其他氨基酸的概率分布。

    参数:
        seq (str): 输入的氨基酸序列。
        k (int): N-Gram 的大小（默认为1，即计算后接一个氨基酸的概率）。

    返回:
        prob_matrix (dict): 一个字典，包含每个氨基酸后接其他氨基酸的概率分布。
    """
    ngram_count = defaultdict(int)
    total = 0
    
    # 统计所有的 N-Gram 频率
    for i in range(len(seq) - k):
        key = (seq[i], seq[i + k])
        ngram_count[key] += 1
        total += 1
    
    # 计算概率
    prob_matrix = {aa: [0.0] * 20 for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    aa_to_index = {aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    
    # 将频率转化为概率
    for (a1, a2), count in ngram_count.items():
        if a1 in 'ACDEFGHIKLMNPQRSTVWY' and a2 in 'ACDEFGHIKLMNPQRSTVWY':
            idx = aa_to_index[a2]
            prob_matrix[a1][idx] = count / total
    
    return prob_matrix
# 读取输入文件
df = pd.read_csv('Bubalus_bubalis_processed_entries_100.csv')
# 使用sequence作为键，label作为值
#label_dict = df['label'].to_dict()  # 字典形式：{行号: label}
node_records = []
edge_records = []
# 处理每个肽链
print("Processing sequences...")
for row_idx, row in tqdm(df.iterrows(), total=len(df)):
    seq = row['sequence']
    length = row['length']
    
    # 根据行号从label_dict获取对应的label值，无需对row进行另行映射
    #label = label_dict.get(row_idx, -1)  # 如果行号不存在则返回 -1，避免 KeyError
    
    # 生成N-Gram统计
    # 注意：请确保 compute_ngram_probs 函数是正确定义的
    ngram_probs = compute_ngram_probs(seq)

    for i, aa in enumerate(seq):
        if aa not in zscale_data:
            continue
        z = zscale_data[aa]
        onehot = one_hot[aa_to_index[aa]]
        position = i / length
        node_feat = z + list(onehot) + [position]  # 只保留 Z-scale 和 One-Hot 特征
        
        # 在每个节点特征中添加label
        #node_records.append([row_idx, i, aa] + node_feat + [label])  # row_idx 作为键
        node_records.append([row_idx, i, aa] + node_feat )  # row_idx 作为键
        # 邻接边（i 到 i+1）
        if i < len(seq)-1:
            aa2 = seq[i+1]
            if aa2 not in zscale_data:
                continue
            blosum_score = get_blosum62_score(aa, aa2)
            edge_feat = [blosum_score] + ngram_probs[aa]
            edge_records.append([row_idx, i, i+1] + edge_feat)  # row_idx作为主键

# === 写出文件 ===
# 包括行号作为键sequence更替列
node_columns = ['row_id', 'node_id', 'amino_acid'] + \
               [f'z{i+1}' for i in range(5)] + \
               [f'onehot_{i+1}' for i in range(20)] + ['position']  # 添加 'label' 列

edge_columns = ['row_id', 'source_node', 'target_node', 'blosum62'] + \
               [f'ngram_{i+1}' for i in range(20)]


# 保存节点和边数据
os.makedirs('output', exist_ok=True)
pd.DataFrame(node_records, columns=node_columns).to_csv('Bubalus_bubalis_processed_entries_100_N.csv', index=False)
pd.DataFrame(edge_records, columns=edge_columns).to_csv('Bubalus_bubalis_processed_entries_100_E.csv', index=False)

print("节点特征和边特征提取完成")