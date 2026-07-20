import torch
import torch.nn as nn

from preprocess import HousePriceData
from model import HousePriceModel


def train_one(corr_threshold, p, weight_decay, epochs=3000, patience=50, lr=0.01):
    torch.manual_seed(42)                          # 재현성 고정
    X_train, X_val, y_train, y_val, X_test, test_ids = \
        HousePriceData(corr_threshold=corr_threshold).prepare()

    model = HousePriceModel(input_dim=X_train.shape[1], p=p)          # ★ dropout
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # ★ weight_decay

    best_val = float("inf"); counter = 0
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad()
        loss = criterion(model(X_train), y_train); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        if val_loss < best_val:
            best_val = val_loss; counter = 0
        else:
            counter += 1
        if counter >= patience:
            break
    return X_train.shape[1], best_val ** 0.5


# (변수 임계값, dropout p, weight_decay, 라벨)
configs = [
    (None, 0.0, 0.0,  "262개, 규제 없음"),
    (None, 0.1, 0.0,  "262개, 규제 있음"),   # ★ 약한 규제: dropout 0.1, weight_decay 0
    (0.3,  0.0, 0.0,  "46개,  규제 없음"),
    (0.3,  0.1, 0.0,  "46개,  규제 있음"),   # ★ 약한 규제: dropout 0.1, weight_decay 0
]

results = []
for t, p, wd, label in configs:
    n, rmse = train_one(t, p, wd)
    results.append((label, n, rmse))
    print(f"{label} | 변수 {n}개 | best val RMSE {rmse:.4f}")

print("\n===== 요약 =====")
for label, n, r in results:
    print(f"  {label:18s} | {r:.4f}")