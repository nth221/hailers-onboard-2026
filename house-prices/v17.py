import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from preprocess import HousePriceData
from model2 import HousePriceModelV2


def train_fold(X_tr, y_tr, X_va, y_va, seed=42,
               batch_size=64, lr=0.005, p=0.1, epochs=3000, patience=200):
    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    model = HousePriceModelV2(input_dim=X_tr.shape[1], p=p)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best, best_state, cnt = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(X_va), y_va).item()
        if vl < best:
            best, best_state, cnt = vl, copy.deepcopy(model.state_dict()), 0
        else:
            cnt += 1
        if cnt >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(X_va)


def cv_mlp(fix_skew, full_ordinal, lr, patience):
    X, y, X_test, test_ids, outlier_idx = HousePriceData(
        remove_outliers=True, fix_skew=fix_skew, full_ordinal=full_ordinal
    ).prepare_full()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        drop = outlier_idx.intersection(X_tr.index)
        X_tr, y_tr = X_tr.drop(index=drop), y_tr.drop(index=drop)

        sc = StandardScaler()
        X_tr_t = torch.tensor(sc.fit_transform(X_tr), dtype=torch.float32)
        X_va_t = torch.tensor(sc.transform(X_va),     dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32).view(-1, 1)
        y_va_t = torch.tensor(y_va.values, dtype=torch.float32).view(-1, 1)

        oof[va_idx] = train_fold(X_tr_t, y_tr_t, X_va_t, y_va_t,
                                 lr=lr, patience=patience).squeeze().numpy()

    return np.sqrt(((oof - y.values) ** 2).mean()), X.shape[1]


# ===== 새 표현(①왜도 + ②서열형)에 맞는 학습률 재탐색 =====
print("새 표현 학습률 재탐색")
print("참조: 기준표현 lr=0.005 → OOF 0.1338 / 새표현 lr=0.005 → OOF 0.2054\n")

for lr in [0.002, 0.001]:
    score, ncol = cv_mlp(True, True, lr=lr, patience=200)
    print(f">>> lr={lr} | 변수 {ncol}개 | OOF {score:.4f}")