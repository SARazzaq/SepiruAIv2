"""
PyTorch deep learning models for tabular data.
TabNet-style MLP and a simple LSTM for sequence prediction.
100% free & open-source (BSD license).
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


# ── Tabular MLP ────────────────────────────────────────────────────────────────

def _build_mlp(input_dim: int, output_dim: int, hidden: list, dropout: float = 0.3):
    """Build a simple MLP classifier/regressor with PyTorch."""
    import torch.nn as nn
    layers = []
    prev = input_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def train_pytorch_tabular(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str = "classification",  # "classification" | "regression"
    epochs: int = 30,
    lr: float = 1e-3,
    hidden: Optional[list] = None,
    progress_cb=None,   # optional callable(epoch, loss)
) -> Tuple[object, list, list]:
    """
    Train a PyTorch MLP on tabular data.
    Returns (trained_model, train_losses, val_losses).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    if hidden is None:
        hidden = [128, 64, 32]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_classes = int(np.max(y_train) + 1) if task == "classification" else 1
    output_dim = n_classes if task == "classification" else 1

    model = _build_mlp(X_train.shape[1], output_dim, hidden).to(device)

    X_tr = torch.FloatTensor(X_train).to(device)
    X_vl = torch.FloatTensor(X_val).to(device)

    if task == "classification":
        y_tr = torch.LongTensor(y_train.astype(int)).to(device)
        y_vl = torch.LongTensor(y_val.astype(int)).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        y_tr = torch.FloatTensor(y_train.astype(float)).unsqueeze(1).to(device)
        y_vl = torch.FloatTensor(y_val.astype(float)).unsqueeze(1).to(device)
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        scheduler.step()
        ep_loss /= len(loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_vl)
            val_loss = criterion(val_pred, y_vl).item()

        train_losses.append(round(ep_loss, 4))
        val_losses.append(round(val_loss, 4))

        if progress_cb:
            progress_cb(epoch + 1, ep_loss)

    return model, train_losses, val_losses


def pytorch_predict(model, X: np.ndarray, task: str = "classification") -> np.ndarray:
    """Run inference with a trained PyTorch model."""
    import torch
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        out = model(X_t)
        if task == "classification":
            return torch.argmax(out, dim=1).cpu().numpy()
        else:
            return out.squeeze(1).cpu().numpy()


def pytorch_score(model, X: np.ndarray, y: np.ndarray, task: str = "classification") -> dict:
    """Compute accuracy (classification) or R2/MAE (regression)."""
    from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
    preds = pytorch_predict(model, X, task)
    if task == "classification":
        return {"accuracy": round(float(accuracy_score(y.astype(int), preds)), 4)}
    else:
        return {
            "r2": round(float(r2_score(y, preds)), 4),
            "mae": round(float(mean_absolute_error(y, preds)), 4),
        }
