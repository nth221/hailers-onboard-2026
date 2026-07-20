# ==========================================================
#  v21.py — 상호작용 특성 검증
#  고정: 표현(왜도보정+서열형확장), 구조 32→8, K-fold 분할
#  변경: add_interactions (곱·비율 특성 7개)
#  ※ v17 교훈 반영 — 특성이 바뀌면 최적 lr도 달라지므로 lr 2가지 측정
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


def cv_mlp(model_cls, lr, p, add_interactions=False):
    X, y, X_test, test_ids, outlier_idx = HousePriceData(
        remove_outliers=True, fix_skew=True, full_ordinal=True,
        add_interactions=add_interactions                 # ★[v21]
    ).prepare_full()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # v15~v20과 동일 분할
    oof = np.zeros(len(X))

    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        drop = outlier_idx.intersection(X_tr.index)         # fold-train에서만 이상치 제거
        X_tr, y_tr = X_tr.drop(index=drop), y_tr.drop(index=drop)

        sc = StandardScaler()                               # fold-train에만 fit
        X_tr_t = torch.tensor(sc.fit_transform(X_tr), dtype=torch.float32)
        X_va_t = torch.tensor(sc.transform(X_va),     dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32).view(-1, 1)
        y_va_t = torch.tensor(y_va.values, dtype=torch.float32).view(-1, 1)

        oof[va_idx] = train_fold(X_tr_t, y_tr_t, X_va_t, y_va_t,
                                 model_cls, lr, p).squeeze().numpy()

    return np.sqrt(((oof - y.values) ** 2).mean()), X.shape[1]


# ===== 비교 =====
MODEL = partial(Flex, hidden=(32, 8))                       # v19 최적 구조 고정

configs = [
    (0.002, False, "기준 (상호작용 없음)"),
    (0.002, True,  "상호작용 추가, lr=0.002"),
    (0.001, True,  "상호작용 추가, lr=0.001"),
]

print("=" * 62)
print("v21: 상호작용 특성 검증 (구조 32→8 고정)")
print("=" * 62)

results = []
for lr, inter, label in configs:
    score, ncol = cv_mlp(MODEL, lr, 0.1, add_interactions=inter)
    results.append((label, ncol, score))
    print(f"\n>>> {label:26s} | 변수 {ncol}개 | OOF {score:.4f}\n")

print("=" * 62)
print(f"{'구성':28s} | 변수  | OOF RMSE")
print("-" * 62)
for label, ncol, score in results:
    print(f"{label:28s} | {ncol:4d}  | {score:.4f}")
print("=" * 62)
print("참조: 현재 최고 OOF 0.1278 / Kaggle 0.13343")