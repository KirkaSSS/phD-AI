import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
#  Neural Network definition
# ============================================================
class NNRegressor(nn.Module):
    """
    3-hidden-layer MLP with BatchNorm + Dropout.
    Architecture mirrors the complexity budget of the CatBoost
    model (depth=3, 250 trees) while staying a pure black-box.
    """
    def __init__(self, input_dim: int):
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
            nn.Dropout(0.1),

            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ============================================================
#  Training helpers
# ============================================================
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS     = 300        # analogous to CatBoost iterations=250
LR         = 0.001      # analogous to learning_rate=0.08
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-3     # analogous to l2_leaf_reg=1.75


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
    tx = torch.tensor(X, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(tx, ty), batch_size=BATCH_SIZE, shuffle=shuffle)


def train_model(X_tr: np.ndarray, y_tr: np.ndarray) -> NNRegressor:
    model = NNRegressor(X_tr.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()
    loader = make_loader(X_tr, y_tr)

    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()
    return model


def predict(model: NNRegressor, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tx = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        return model(tx).cpu().numpy()


def get_model(X_tr: np.ndarray, y_tr: np.ndarray) -> NNRegressor:
    """Drop-in replacement for CatBoost's get_model() + fit()."""
    return train_model(X_tr, y_tr)


# ============================================================
#  Load data  (identical to white-box script)
# ============================================================
#train_df = pd.read_excel("Cr3_dgb_training_set_pročišćen_proširen.xlsx")

train_df = pd.read_excel(r"C:\Users\Korisnik\PycharmProjects\DqBpredictor-main\Cr3_dgb_training_set_pročišćen_proširen.xlsx")


X, y = train_df.iloc[:, 2:20].values, train_df.iloc[:, 1].values
#predict_df = pd.read_excel("To_predict.xlsx")
predict_df = pd.read_excel(r"C:\Users\Korisnik\PycharmProjects\DqBpredictor-main\To_predict.xlsx")

X_new, formulas = predict_df.drop(columns=["Formula"]).values, predict_df["Formula"].values

print("🔄 Neural Network model is running to find the best random state...")

# ============================================================
#  Find best random state  (identical logic)
# ============================================================
best_r2, best_state = -np.inf, None

for rs in sorted(set(range(5, 101, 5)).union(range(5, 101, 7))):
    r2s = []
    for tr_idx, te_idx in KFold(n_splits=10, shuffle=True, random_state=rs).split(X):
        sc = StandardScaler().fit(X[tr_idx])
        model = get_model(sc.transform(X[tr_idx]), y[tr_idx])
        r2s.append(r2_score(y[te_idx], predict(model, sc.transform(X[te_idx]))))
    mean_r2 = np.mean(r2s)
    if mean_r2 > best_r2:
        best_r2, best_state = mean_r2, rs

print(f"✅ Best random state = {best_state}")

# ============================================================
#  Final 10-fold CV  (identical structure)
# ============================================================
kf = KFold(n_splits=10, shuffle=True, random_state=best_state)
y_true, y_pred, r2_scores, fold_preds_new = [], [], [], []
train_preds = [[] for _ in range(len(y))]

for tr_idx, te_idx in kf.split(X):
    sc = StandardScaler().fit(X[tr_idx])
    model = get_model(sc.transform(X[tr_idx]), y[tr_idx])
    preds = predict(model, sc.transform(X[te_idx]))
    y_true.extend(y[te_idx])
    y_pred.extend(preds)
    r2_scores.append(r2_score(y[te_idx], preds))
    for idx, p in zip(te_idx, preds):
        train_preds[idx].append(p)
    fold_preds_new.append(predict(model, sc.transform(X_new)))

# ============================================================
#  Final model trained on ALL data
# ============================================================
sc_full = StandardScaler().fit(X)
final_model = get_model(sc_full.transform(X), y)
final_preds = predict(final_model, sc_full.transform(X_new))
uncertainty  = np.std(np.array(fold_preds_new), axis=0)

# ============================================================
#  Save to Excel  (identical output format)
# ============================================================
#pd.DataFrame({...}).to_excel(r"C:\Users\Korisnik\Downloads\final_prediction_with_uncertainty_NN.xlsx", index=False)

(pd.DataFrame({
    "Formula":        formulas,
    "Predicted Dq/B": final_preds,
    "Uncertainty":    uncertainty,
}).to_excel(r"C:\Users\Korisnik\PycharmProjects\DqBpredictor-main\final_prediction_with_uncertainty_NN.xlsx", index=False))

 #("final_prediction_with_uncertainty_NN.xlsx", index=False))

# ============================================================
#  CV Metrics  (identical reporting)
# ============================================================
final_r2 = r2_score(y_true, y_pred)
mae      = mean_absolute_error(y_true, y_pred)

print("\n📊 R² scores per fold (best random_state):")
for i, r2 in enumerate(r2_scores, 1):
    print(f"Fold {i}: R² = {r2:.4f}")
print(f"\n✅ Final Combined R² = {final_r2:.4f}")
print(f"📉 MAE = {mae:.4f}")

print("\n📄 Final Predictions with Uncertainty:")
print("Formula                           Predicted Dq/B     Uncertainty")
for f, p, u in zip(formulas, final_preds, uncertainty):
    print(f"{f:<32} {p:>10.4f}         {u:.4f}")

# ============================================================
#  Parity Plot  (identical)
# ============================================================
plt.figure(figsize=(7, 7))
plt.scatter(y, [np.mean(p) for p in train_preds], alpha=0.7)
plt.plot([min(y), max(y)], [min(y), max(y)], "r--", lw=2)
plt.xlabel("True Dq/B")
plt.ylabel("Predicted Dq/B")
plt.title(
    f"Parity Plot – Neural Network (Best Random State: {best_state})\n"
    f"R² = {final_r2:.4f}, MAE = {mae:.4f}"
)
plt.grid(True)
plt.tight_layout()
plt.savefig("parity_plot_NN.png", dpi=150)
plt.show()
print("✅ Saved 'final_prediction_with_uncertainty_NN.xlsx'")
