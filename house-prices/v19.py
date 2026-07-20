# ==========================================================
#  v19.py — 모델 용량 축소 탐색
#  v18 결과: 용량 확대(10만 파라미터)는 악화 → "작을수록 유리" 방향 확인
#  목적: 64→16보다 작은 구성에서 최소점이 있는지 탐색
#  고정: 표현(224개), K-fold 분할, lr=0.002, dropout=0.1
# ==========================================================

import copy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from preprocess import HousePriceData
from model_flex import HousePriceModelFlex as Flex


def train_fold(X_tr, y_tr, X_va, y_va, model_cls, lr, p,
               seed=42, batch_size=64, epochs=3000, patience=200):
    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    model = model_cls(input_dim=X_tr.shape[1], p=p)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best, best_state, cnt = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

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


def cv_mlp(model_cls, lr, p, fix_skew=True, full_ordinal=True):
    X, y, X_test, test_ids, outlier_idx = HousePriceData(
        remove_outliers=True, fix_skew=fix_skew, full_ordinal=full_ordinal
    ).prepare_full()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)   # v15~v18과 동일 분할
    oof = np.zeros(len(X))

    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        drop = outlier_idx.intersection(X_tr.index)          # fold-train에서만 이상치 제거
        X_tr, y_tr = X_tr.drop(index=drop), y_tr.drop(index=drop)

        sc = StandardScaler()                                # fold-train에만 fit
        X_tr_t = torch.tensor(sc.fit_transform(X_tr), dtype=torch.float32)
        X_va_t = torch.tensor(sc.transform(X_va),     dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32).view(-1, 1)
        y_va_t = torch.tensor(y_va.values, dtype=torch.float32).view(-1, 1)

        oof[va_idx] = train_fold(X_tr_t, y_tr_t, X_va_t, y_va_t,
                                 model_cls, lr, p).squeeze().numpy()

    return np.sqrt(((oof - y.values) ** 2).mean()), X.shape[1]


# ===== 은닉층 크기만 변경 =====
configs = [
    (partial(Flex, hidden=(128, 32)), 0.002, 0.1, "128→32"),
    (partial(Flex, hidden=(64, 16)),  0.002, 0.1, "64→16   [기준=model2]"),
    (partial(Flex, hidden=(32, 8)),   0.002, 0.1, "32→8"),
    (partial(Flex, hidden=(16, 4)),   0.002, 0.1, "16→4"),
]

print("=" * 60)
print("v19: 모델 용량 축소 탐색 (표현 224개, lr=0.002 고정)")
print("=" * 60)

results = []
for cls, lr, p, label in configs:
    score, ncol = cv_mlp(cls, lr, p)
    results.append((label, score))
    print(f"\n>>> {label:22s} | OOF {score:.4f}\n")

print("=" * 60)
print(f"{'은닉층':24s} | OOF RMSE")
print("-" * 60)
for label, score in results:
    print(f"{label:24s} | {score:.4f}")
print("=" * 60)
print("참조: 64→16 기준 0.1299 / model3(256→128→64) 0.1325 / 튜닝 트리 0.1304")