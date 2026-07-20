# ==========================================================
#  v18.py — ② 모델 용량 재검토
#  기준: 새 표현(왜도보정+서열형확장, 224개) + model2 + lr=0.002 → OOF 0.1299
#  목적: model2(64→16)가 적정 크기인지, 더 큰 model3가 나은지 비교
#        ※ model2는 풀배치 시절(v7)의 유산이라 미니배치 정상화 후 재검토 필요
# ==========================================================

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from preprocess import HousePriceData
from model2 import HousePriceModelV2
from model3 import HousePriceModelV3


def train_fold(X_tr, y_tr, X_va, y_va, model_cls, lr, p,
               seed=42, batch_size=64, epochs=3000, patience=200):
    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    model = model_cls(input_dim=X_tr.shape[1], p=p)        # ★ 모델 클래스를 인자로 받음
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

    kf = KFold(n_splits=5, shuffle=True, random_state=42)   # v15~v17과 동일 분할
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


# ===== 비교할 조합 =====
configs = [
    (HousePriceModelV2, 0.002, 0.1, "model2 (64→16)      [기준]"),
    (HousePriceModelV3, 0.002, 0.2, "model3 (256→128→64) lr=0.002"),
    (HousePriceModelV3, 0.001, 0.2, "model3 (256→128→64) lr=0.001"),
]

print("=" * 60)
print("② 모델 용량 재검토 (표현: 왜도보정+서열형확장, 224개)")
print("=" * 60)

results = []
for cls, lr, p, label in configs:
    score, ncol = cv_mlp(cls, lr, p)
    results.append((label, score))
    print(f"\n>>> {label:30s} | OOF {score:.4f}\n")

print("=" * 60)
print(f"{'구성':32s} | OOF RMSE")
print("-" * 60)
for label, score in results:
    print(f"{label:32s} | {score:.4f}")
print("=" * 60)
print("참조: model2 기준 0.1299 / 튜닝 트리 0.1304 / 기존 표현 MLP 0.1338")