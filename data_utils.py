import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split

# 特殊标记
GO = "<go>"
EOS = "<eos>"


def build_vocabulary(smiles_list):
    """根据SMILES列表构建词汇表与映射字典"""
    chars = set("".join(smiles_list))
    chars.update([GO, EOS])
    chars = sorted(chars)
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


def vectorize_int(smiles_list, char2idx, max_len):
    """
    返回整数序列 (B, T)
    """
    indices = np.zeros((len(smiles_list), max_len), dtype=np.int64)
    for i, smi in enumerate(smiles_list):
        idx_seq = [char2idx[GO]]
        idx_seq.extend(char2idx[c] for c in smi)
        idx_seq.append(char2idx[EOS])
        # 截断或填充
        idx_seq = idx_seq[:max_len] + [char2idx[EOS]] * max(0, max_len - len(idx_seq))
        indices[i] = idx_seq
    return torch.tensor(indices)


def load_and_split_data(smi_path, train_ratio=0.8, sample_size=None):
    data = pd.read_csv(smi_path, sep="\t", header=None, names=["smiles", "No", "Int"])
    smiles = data["smiles"].tolist()
    if sample_size:
        smiles = smiles[:sample_size]

    # T
    max_len = max(len(s) for s in smiles) + 2  # <go> + chars + <eos>

    char2idx, idx2char = build_vocabulary(smiles)
    indices = vectorize_int(smiles, char2idx, max_len)
    dataset = TensorDataset(indices)
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    train_ds, test_ds = random_split(
        dataset, [n_train, n_total - n_train],
        generator=torch.Generator().manual_seed(42))
    return train_ds, test_ds, char2idx, idx2char, max_len, len(char2idx)


def indices_to_smiles(indices, idx2char):
    """将索引序列转为SMILES字符串"""
    chars = []
    for idx in indices:
        c = idx2char[idx.item()]
        if c == EOS:
            break
        if c != GO:
            chars.append(c)
    return "".join(chars)
