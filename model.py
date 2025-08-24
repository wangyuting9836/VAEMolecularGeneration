import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # 相应的KL散度及其关于 $\ln \sigma^2$ 的导数，以及重参数化技巧，均可轻松适配这一修改。
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
