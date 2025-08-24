# 用 VAE 从零搭一个「分子生成器」

**Author:** Yuting Wang

**Date:** 2025-08-24

**Link:** https://zhuanlan.zhihu.com/p/1943004826934935811



目录

收起

0\. 为什么选 VAE？

1\. 数据集准备

3\. 模型

4\. 训练验证

5\. 推理生成分子

把字符组成的 SMILES 映射到 256 维潜空间的潜向量（化学指纹），然后从潜空间随机采样潜向量，对潜向量进行 decode 就能批量产出从未见过的分子。

完整代码已上传 GitHub。

[VAEMolecularGeneration​github.com/wangyuting9836/VAEMolecularGeneration](https://link.zhihu.com/?target=https%3A//github.com/wangyuting9836/VAEMolecularGeneration)

## 0\. 为什么选 VAE？

-   **编码器** → 根据嵌入矩阵生成成 **高斯分布**
-   **重参数** → 随机采样潜在向量 **z**
-   **解码器** → 根据潜向量 **z** 一次性还原整条 SMILES。

关于 VAE 的详细介绍可以参考下面链接，这个介绍的比较清楚。

[The Mathematics of Variational Auto-Encoders​davidstutz.de/the-mathematics-of-variational-auto-encoders/](https://link.zhihu.com/?target=https%3A//davidstutz.de/the-mathematics-of-variational-auto-encoders/)

在上面的博客中存在一处错误，除了这个错误，其他讲的很清楚。博客中把求两个高斯分布 Kullback-Leibler 散度的公式写成了下面形式

$\text{KL}(\mathcal{N}(z_i ; \mu_{1,i}, \sigma_{1,i}^2)|\mathcal{N}(z_i ; \mu_{2,i},\sigma_{2,i}^2)) = \frac{1}{2}\ln\frac{\sigma_{2,i}}{\sigma_{1,i}} + \frac{\sigma_{1,i}^2}{2\sigma_{2,i}^2} + \frac{(\mu_{1,i} - \mu_{2,i})^2}{2 \sigma_{2,i}^2} - \frac{1}{2}$ ，

导致推出 $\text{KL}(p(z_i | y) | p(z_i)) = - \frac{1}{2}\ln \sigma_i + \frac{1}{2} \sigma_i^2 + \frac{1}{2} \mu_i^2 - \frac{1}{2}$ ，第一项前面多了个$\frac{1}{2}$。

正确的应该是下面形式，

$\text{KL}(\mathcal{N}(z_i ; \mu_{1,i}, \sigma_{1,i}^2)|\mathcal{N}(z_i ; \mu_{2,i},\sigma_{2,i}^2)) = \ln\frac{\sigma_{2,i}}{\sigma_{1,i}} + \frac{\sigma_{1,i}^2}{2\sigma_{2,i}^2} + \frac{(\mu_{1,i} - \mu_{2,i})^2}{2 \sigma_{2,i}^2} - \frac{1}{2}$ ，

可推出 $\text{KL}(p(z_i | y) | p(z_i)) = - \ln \sigma_i + \frac{1}{2} \sigma_i^2 + \frac{1}{2} \mu_i^2 - \frac{1}{2}= -\frac{1}{2} \ln \sigma_i^2 + \frac{1}{2} \sigma_i^2 + \frac{1}{2} \mu_i^2 - \frac{1}{2}$，这对应损函数中的如下代码。

```python
kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
```

关于 VAE 还可以参考下面的链接，这个是中文。

[](https://link.zhihu.com/?target=https%3A//fenghz.github.io/Variational-AutoEncoder/)

## 1\. 数据集准备

这里使用 GDB Database。

GDB-11 数据库通过应用简单的化学稳定性与合成可行性规则，枚举了所有由最多 11 个碳、氮、氧、氟原子组成的有机小分子。

GDB-13 数据库通过应用简单的化学稳定性与合成可行性规则，枚举了所有由最多 13 个碳、氮、氧、硫、氯原子组成的有机小分子。该数据库拥有 977,468,314 个分子结构，是迄今为止全球最大的公开有机小分子数据库。

链接如下。

[](https://link.zhihu.com/?target=https%3A//gdb.unibe.ch/downloads/)

2\. 根据数据集构建词汇表， 把 SMILES 变成整数序列。对于整数序列在 VAE 中需进行 Embedding。

```python
train_ds, test_ds, char2idx, idx2char, max_len, vocab_size = load_and_split_data("gdb11/gdb11_size08.smi", sample_size=None)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
```

函数实现如下。

```python
# 特殊标记
GO = "<go>"
EOS = "<eos>"


def build_vocabulary(smiles_list):
    """根据 SMILES 列表构建词汇表与映射字典"""
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

    # 2021 年 12 月 17 日
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
    """将索引序列转为 SMILES 字符串"""
    chars = []
    for idx in indices:
        c = idx2char[idx.item()]
        if c == EOS:
            break
        if c != GO:
            chars.append(c)
    return "".join(chars)
```

## 3\. 模型

```python
model = VAE(
    vocab_size=vocab_size,
    max_len=max_len,
    embed_dim=64,
    latent_dim=256
).to(device)
```

具体代码如下，首先需要对整数序列进行 Embedding，Encoder 和 Decoder 用全连接。

```python
class VAE(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 max_len: int,
                 embed_dim: int = 64,
                 hidden_dim: int = 512,
                 latent_dim: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.latent_dim = latent_dim

        # 1. Embedding 层
        self.embed = nn.Embedding(vocab_size, embed_dim)

        # 2. 编码器：flatten -> 512 -> 512 -> μ/logvar
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),  # (B, max_len * embed_dim)
            nn.Linear(max_len * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mu = nn.Linear(hidden_dim, latent_dim)
        # 在实践中，直接让编码器预测 $\sigma^2$ 会产生问题，因为方差 $\sigma^2$ 不可为负值。
        # 因此，我们可以让编码器改为预测对数方差，即 $\ln \sigma^2$。
        # 这确保了方差 $\sigma^2 = \exp(\ln \sigma^2)$ 始终为正数。
        # 相应的 KL 散度及其关于 $\ln \sigma^2$ 的导数，以及重参数化技巧，均可轻松适配这一修改。
        self.log_var = nn.Linear(hidden_dim, latent_dim)

        # 3. 解码器：latent -> 512 -> 512 -> 重构 logits
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_len * vocab_size)
        )

    def encode(self, x):
        """
        x: (B, T)
        返回:
            mu: (B, latent_dim)
            log_var: (B, latent_dim)
        """
        e = self.embed(x)  # (B, T, embed_dim)
        h = self.encoder_fc(e)  # (B, hidden_dim)
        mu, log_var = self.mu(h), self.log_var(h) # (B, latent_dim)
        return mu, log_var

    @staticmethod
    def re_parameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        z: (B, latent_dim)
        返回:
            logits (B, T, V)
        """
        logits = self.decoder_fc(z)  # (B, T*V)
        logits = logits.view(-1, self.max_len, self.vocab_size)
        return logits

    def forward(self, x):
        """
        x: (B, T)
        返回:
            logits: (B, T, V)
            mu: (B, latent_dim)
            log_var: (B, latent_dim)
        """
        mu, log_var = self.encode(x)
        z = self.re_parameterize(mu, log_var)
        logits = self.decode(z)
        return logits, mu, log_var
```

## 4\. 训练验证

-   **kl\_weight**：训练对 VAE 损失函数中的 KL 散度项进行权重调整
-   **ReduceLROnPlateau**：验证集 5 个 epoch 不下降 → 学习率 ×0.5。
-   **Early Stopping**：验证集 10 个 epoch 不下降 → 直接停。

```python
train(
    model,
    train_loader,
    test_loader,
    device=device,
    epochs=1000,
    lr=1e-3,
    save_path="best_model.pth",
)
```

具体实现如下。

```python
def loss_function(logits, target, mu, log_var, kl_weight=1.0):
    """
    logits: (B, T, V)
    target: (B, T) 整数序列
    mu: (B, latent_dim)
    log_var: (B, latent_dim)
    """
    recon_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_weight * kl_loss


def train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=50,
        lr=1e-3,
        save_path="best_model.pth",
        patience=20,
        kl_weight=1.0
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 2, min_lr=1e-6)

    best_val = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            logits, mu, log_var = model(x)
            loss = loss_function(logits, x, mu, log_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, in val_loader:
                x = x.to(device)
                logits, mu, log_var = model(x)
                loss = loss_function(logits, x, mu, log_var, kl_weight)
                val_loss += loss.item()
        avg_val = val_loss / len(val_loader)

        scheduler.step(avg_val)
        print(
            f"Epoch {epoch + 1}/{epochs}  "
            f"Train loss: {avg_train:.4f}  Val loss: {avg_val:.4f}"
        )

        # 保存最佳模型
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
```

## 5\. 推理生成分子

随机从潜空间采样潜向量，根据潜向量由解码器生成分子，然后可视化

```python
z_rand = torch.randn(1, 256, device=device)
smi = generate_from_latent(model, z_rand, idx2char) # 随机从潜在空间采样
print("Random generation:", smi)
visualize_one_smiles(smi) # 可视化
```

具体实现如下。

```python
@torch.no_grad()
def generate_from_latent(model, z, idx2char):
    model.eval()
    logits = model.decode(z.unsqueeze(0))[0]  # (T, V)
    indices = logits.argmax(dim=-1)  # (T,)
    return indices_to_smiles(indices.cpu(), idx2char)
```

生成的分子如下。

![](https://pica.zhimg.com/v2-0ca5c91122b2887a1650121cb08a3f4e_1440w.jpg)
