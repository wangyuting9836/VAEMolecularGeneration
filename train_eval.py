import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F


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
