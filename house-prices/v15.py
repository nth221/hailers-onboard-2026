import copy
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from preprocess import HousePriceData
from model2 import HousePriceModelV2

# ===== 원본 상태로 데이터 받기 =====
X, y, X_test_raw, test_ids, outlier_idx = \
    HousePriceData(remove_outliers=True).prepare_full()


def train_fold(X_tr, y_tr, X_va, y_va, X_te, seed=42,
               batch_size=64, lr=0.005, p=0.1, epochs=2000, patience=100):
    torch.manual_seed(seed)
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    model = HousePriceModelV2(input_dim=X_tr.shape[1], p=p)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val, best_state, counter = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_va), y_va).item()
        if val_loss < best_val:
            best_val, best_state, counter = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            counter += 1
        if counter >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return best_val ** 0.5, model(X_va), model(X_te)


# ===== 5-fold 교차검증 =====
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(X))                                    # out-of-fold 예측 저장
test_preds, fold_rmses = [], []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # 이상치는 fold-train에서만 제거 (검증 fold는 손대지 않음)
    drop = outlier_idx.intersection(X_tr.index)
    X_tr, y_tr = X_tr.drop(index=drop), y_tr.drop(index=drop)

    # 스케일링: fold-train에만 fit (누수 방지)
    scaler = StandardScaler()
    X_tr_s = torch.tensor(scaler.fit_transform(X_tr), dtype=torch.float32)
    X_va_s = torch.tensor(scaler.transform(X_va),     dtype=torch.float32)
    X_te_s = torch.tensor(scaler.transform(X_test_raw), dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32).view(-1, 1)
    y_va_t = torch.tensor(y_va.values, dtype=torch.float32).view(-1, 1)

    rmse, va_pred, te_pred = train_fold(X_tr_s, y_tr_t, X_va_s, y_va_t, X_te_s)
    oof_pred[va_idx] = va_pred.squeeze().numpy()
    test_preds.append(te_pred)
    fold_rmses.append(rmse)
    print(f"fold {fold} | train {len(X_tr)} / val {len(X_va)} | val RMSE {rmse:.4f}")

# ===== 결과 =====
oof_rmse = np.sqrt(((oof_pred - y.values) ** 2).mean())        # 전체 1460개 기준
print("\n===== 결과 =====")
print(f"fold별 RMSE   : {np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.4f}")
print(f"★ OOF RMSE    : {oof_rmse:.4f}   (전체 {len(X)}개 평가)")

# ===== fold 앙상블 제출 =====
test_avg = torch.stack(test_preds).mean(dim=0)
pred = torch.expm1(test_avg).squeeze().numpy()
submission = pd.DataFrame({"Id": test_ids, "SalePrice": pred})
submission.to_csv("submission.csv", index=False)
print("submission.csv 저장 완료:", submission.shape)