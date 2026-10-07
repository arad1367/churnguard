"""A small PyTorch MLP for tabular churn prediction, wrapped so it behaves
like any scikit-learn classifier (fit / predict_proba).

Because it follows the scikit-learn interface, it plugs straight into our
preprocessing Pipeline, cross-validation, and metrics from Phase 4.
"""
import copy

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ChurnMLP(nn.Module):
    """Feed-forward network: [Linear → ReLU → Dropout] × N → Linear(1)."""

    def __init__(self, n_features: int, hidden_sizes: tuple[int, ...] = (64, 32),
                 dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        in_size = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.ReLU(), nn.Dropout(dropout)]
            in_size = h
        layers.append(nn.Linear(in_size, 1))   # one output: the churn logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)          # shape (batch,) of logits


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """scikit-learn compatible wrapper with early stopping on a validation split.

    All hyperparameters are __init__ arguments stored unchanged, which is what
    scikit-learn needs to clone the estimator (e.g. in cross-validation).
    """

    def __init__(self, hidden_sizes=(64, 32), dropout=0.2, lr=1e-3, weight_decay=1e-4,
                 batch_size=64, max_epochs=200, patience=15, use_pos_weight=False,
                 val_fraction=0.15, random_state=42):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience            # None = no early stopping
        self.use_pos_weight = use_pos_weight
        self.val_fraction = val_fraction
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        # Internal validation split, used ONLY for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=self.val_fraction, stratify=y, random_state=self.random_state
        )
        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=self.batch_size,
            shuffle=True,              # new random batch order each epoch
            generator=generator,
        )
        X_val_t, y_val_t = torch.from_numpy(X_val), torch.from_numpy(y_val)

        self.model_ = ChurnMLP(self.n_features_in_, tuple(self.hidden_sizes), self.dropout)
        pos_weight = (
            torch.tensor((y_tr == 0).sum() / (y_tr == 1).sum()) if self.use_pos_weight else None
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.history_ = {"train_loss": [], "val_loss": [], "val_pr_auc": []}
        best_val_loss, best_state, epochs_without_improvement = float("inf"), None, 0
        self.best_epoch_ = 0

        for epoch in range(1, self.max_epochs + 1):
            # ---- training ----
            self.model_.train()                      # dropout ON
            running_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()                # 1. clear old gradients
                logits = self.model_(xb)             # 2. forward pass
                loss = loss_fn(logits, yb)           # 3. compute loss
                loss.backward()                      # 4. backpropagation
                optimizer.step()                     # 5. update weights
                running_loss += loss.item() * len(xb)
            train_loss = running_loss / len(X_tr)

            # ---- validation ----
            self.model_.eval()                       # dropout OFF
            with torch.no_grad():                    # no gradients needed
                val_logits = self.model_(X_val_t)
                val_loss = loss_fn(val_logits, y_val_t).item()
                val_pr_auc = average_precision_score(y_val, torch.sigmoid(val_logits).numpy())

            self.history_["train_loss"].append(train_loss)
            self.history_["val_loss"].append(val_loss)
            self.history_["val_pr_auc"].append(val_pr_auc)

            # ---- early stopping bookkeeping ----
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                self.best_epoch_ = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if self.patience is not None and epochs_without_improvement >= self.patience:
                    break

        self.n_epochs_ = epoch
        if self.patience is not None and best_state is not None:
            self.model_.load_state_dict(best_state)  # restore the best weights
        return self

    def predict_proba(self, X) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(np.asarray(X, dtype=np.float32)))
            p = torch.sigmoid(logits).numpy()
        return np.column_stack([1 - p, p])           # scikit-learn format: [P(0), P(1)]

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)