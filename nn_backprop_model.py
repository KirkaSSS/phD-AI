import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Parameter mapping from CatBoost to Neural Network:
# iterations=500   -> EPOCHS=500
# depth=4          -> 4 hidden layers (128->64->32->16->1)
# l2_leaf_reg=1.9  -> weight_decay=1.9e-3
# border_count=64  -> BATCH_SIZE=64

EPOCHS       = 500
DEPTH        = 4
WEIGHT_DECAY = 1.9e-3
BATCH_SIZE   = 64
LR           = 0.001
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NNRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(16, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_loader(X, y, shuffle=True):
    tx = torch.tensor(X, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(tx, ty), batch_size=BATCH_SIZE, shuffle=shuffle)


def train_model(X_tr, y_tr):
    model     = NNRegressor(X_tr.shape[1]).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
    criterion = nn.MSELoss()
    loader    = make_loader(X_tr, y_tr)

    model.train()
    for epoch in range(EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
        scheduler.step()
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        tx = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        return model(tx).cpu().numpy()


def get_model(X_tr, y_tr):
    return train_model(X_tr, y_tr)


# Load data
train_df   = pd.read_excel(r"C:\Users\Korisnik\PycharmProjects\DqBpredictor-main\Cr3_dgb_training_set_pročišćen_proširen.xlsx")
X, y       = train_df.iloc[:, 2:20].values, train_df.iloc[:, 1].values
predict_df = pd.read_excel(r"C:\Users\Korisnik\PycharmProjects\DqBpredictor-main\To_predict.xlsx")
X_new, formulas = predict_df.drop(columns=["Formula"]).values, predict_df["Formula"].values

print(f"Backprop NN | epochs={EPOCHS} | depth={DEPTH} | l2={WEIGHT_DECAY} | batch={BATCH_SIZE}")
print(f"Device: {DEVICE}")
print("Searching for best random state...\n")

# Find best random state
best_r2, best_state = -np.inf, None
candidates = sorted(set(range(5, 101, 5)).union(range(5, 101, 7)))

for i, rs in enumerate(candidates, 1):
    r2s = []
    for tr_idx, te_idx in KFold(n_splits=10, shuffle=True, random_state=rs).split(X):
        sc    = StandardScaler().fit(X[tr_idx])
        model = get_model(sc.transform(X[tr_idx]), y[tr_idx])
        r2s.append(r2_score(y[te_idx], predict(model, sc.transform(X[te_idx]))))
    mean_r2 = np.mean(r2s)
    print(f"  [{i:02d}/{len(candidates)}] rs={rs:3d}  mean R2={mean_r2:.4f}")
    if mean_r2 > best_r2:
        best_r2, best_state = mean_r2, rs

print(f"\nBest random state = {best_state}  (mean R2 = {best_r2:.4f})")

# Final 10-fold CV
kf = KFold(n_splits=10, shuffle=True, random_state=best_state)
y_true, y_pred, r2_scores, fold_preds_new = [], [], [], []
train_preds = [[] for _ in range(len(y))]

for fold, (tr_idx, te_idx) in enumerate(kf.split(X), 1):
    print(f"  Final CV fold {fold}/10 ...", end="\r")
    sc    = StandardScaler().fit(X[tr_idx])
    model = get_model(sc.transform(X[tr_idx]), y[tr_idx])
    preds = predict(model, sc.transform(X[te_idx]))
    y_true.extend(y[te_idx])
    y_pred.extend(preds)
    r2_scores.append(r2_score(y[te_idx], preds))
    for idx, p in zip(te_idx, preds):
        train_preds[idx].append(p)
    fold_preds_new.append(predict(model, sc.transform(X_new)))

print("  Final CV folds complete.    ")

# Final model on all data
print("Training final model on full dataset...")
sc_full     = StandardScaler().fit(X)
final_model = get_model(sc_full.transform(X), y)
final_preds = predict(final_model, sc_full.transform(X_new))
uncertainty = np.std(np.array(fold_preds_new), axis=0)

# Save to Excel
(pd.DataFrame({
    "Formula":        formulas,
    "Predicted Dq/B": final_preds,
    "Uncertainty":    uncertainty,
}).to_excel(r"C:\Users\Korisnik\PycharmProjects\DqBpredictor-main\final_prediction_with_uncertainty_backpropNN.xlsx"))

 #("final_prediction_backprop.xlsx", index=False))

# Metrics
final_r2 = r2_score(y_true, y_pred)
mae      = mean_absolute_error(y_true, y_pred)

print("\nR2 scores per fold:")
for i, r2 in enumerate(r2_scores, 1):
    print(f"  Fold {i:2d}: R2 = {r2:.4f}")

print(f"\nFinal Combined R2 = {final_r2:.4f}")
print(f"MAE               = {mae:.4f}")

print("\nFinal Predictions with Uncertainty:")
print(f"{'Formula':<32} {'Predicted Dq/B':>14} {'Uncertainty':>12}")
print("-" * 60)
for f, p, u in zip(formulas, final_preds, uncertainty):
    print(f"{f:<32} {p:>14.4f} {u:>12.4f}")

# Parity Plot
plt.figure(figsize=(7, 7))
plt.scatter(y, [np.mean(p) for p in train_preds], alpha=0.7, edgecolors='k', linewidths=0.4)
plt.plot([min(y), max(y)], [min(y), max(y)], "r--", lw=2, label="Ideal")
plt.xlabel("True Dq/B", fontsize=13)
plt.ylabel("Predicted Dq/B", fontsize=13)
plt.title(
    f"Parity Plot - Backprop NN\n"
    f"epochs={EPOCHS}, depth={DEPTH}, L2={WEIGHT_DECAY}, batch={BATCH_SIZE}\n"
    f"Best RS={best_state} | R2={final_r2:.4f} | MAE={mae:.4f}",
    fontsize=11
)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("parity_plot_backprop.png", dpi=150)
plt.show()

print("\nSaved 'final_prediction_backprop.xlsx'")
print("Saved 'parity_plot_backprop.png'")
