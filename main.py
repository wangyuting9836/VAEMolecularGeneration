import random

import torch
from torch.utils.data import DataLoader
from data_utils import load_and_split_data
from model import VAE
from train_eval import train
from generate import generate_from_latent, interpolate, visualize_smiles, visualize_one_smiles

# 1. 数据
train_ds, test_ds, char2idx, idx2char, max_len, vocab_size = load_and_split_data("gdb11/gdb11_size08.smi", sample_size=None)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 模型
model = VAE(
    vocab_size=vocab_size,
    max_len=max_len,
    embed_dim=64,
    latent_dim=256
).to(device)

# 3. 训练
train(
    model,
    train_loader,
    test_loader,
    device=device,
    epochs=1000,
    lr=1e-3,
    save_path="best_model.pth",
)

# 4. 加载最佳模型并生成分子
model.load_state_dict(torch.load("best_model.pth", weights_only=True, map_location=device))

# 4.1 随机从潜在空间采样
z_rand = torch.randn(1, 256, device=device)
smi = generate_from_latent(model, z_rand, idx2char)
print("Random generation:", smi)
visualize_one_smiles(smi)

# 4.2 潜在空间插值
with torch.no_grad():
    idx1, idx2 = random.sample(range(len(test_ds)), 2)
    x1, = test_ds[idx1]
    x2, = test_ds[idx2]
    x1 = x1.unsqueeze(0).to(device)
    x2 = x2.unsqueeze(0).to(device)
    mu1, log_var1 = model.encode(x1)
    z1 = model.re_parameterize(mu1, log_var1)
    mu2, log_var2 = model.encode(x2)
    z2 = model.re_parameterize(mu2, log_var2)

inter_smiles = interpolate(model, z1.squeeze(0), z2.squeeze(0), n_steps=10, idx2char=idx2char)
print("Interpolation smiles:", inter_smiles)
visualize_smiles(inter_smiles)
