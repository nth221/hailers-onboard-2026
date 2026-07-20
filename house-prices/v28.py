# ==========================================================
#  v28.py — 손실 함수 교체 (MSE vs Huber)
#  가설: MSE는 오차를 제곱해 소수 이상 샘플이 그래디언트를 지배한다.
#        Huber는 delta 초과 오차를 선형 처리해 영향력에 상한을 둔다.
#  고정: 임베딩 dim_mult=1, 구조 32→8, lr=0.002, seed 42, fold_seed 42
#  변경: 손실 함수 (MSE / Huber delta 3종)
#  ※ 빠른 스크리닝 — 유망하면 시드 늘려 재검증
# ==========================================================

import copy
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from preprocess import HousePriceData
from model_embed import HousePriceEmbedModel

X_num, X_cat, cards, y, X_num_te, X_cat_te, test_ids, outlier_idx = HousePriceData(
    remove_outliers=True, fix_skew=True, full_ordinal=True
).prepare_embed()

KF = KFold(n_splits=5, shuffle=True, random_state=42)
EVAL = nn.MSELoss()          # ★ 평가는 항상 MSE (지표가 RMSE이므로)


def run(train_crit, lr=0.002, seed=42, dim_mult=1,
        hidden=(32, 8), p=0.1, epochs=3000, patience=200):
    """train_crit: 학습에 쓸 손실 함수 (평가는 항상 MSE)"""
    oof = np.zeros(len(X_num))
    test_preds = []

    for tr_idx, va_idx in KF.split(X_num):
        Xn_tr, Xn_va = X_num.iloc[tr_idx], X_num.iloc[va_idx]
        Xc_tr, Xc_va = X_cat.iloc[tr_idx], X_cat.iloc[va_idx]
        y_tr,  y_va  = y.iloc[tr_idx],     y.iloc[va_idx]

        drop = outlier_idx.intersection(Xn_tr.index)
        Xn_tr = Xn_tr.drop(index=drop)
        Xc_tr = Xc_tr.drop(index=drop)
        y_tr  = y_tr.drop(index=drop)

        sc = StandardScaler()
        Xn_tr_t = torch.tensor(sc.fit_transform(Xn_tr), dtype=torch.float32)
        Xn_va_t = torch.tensor(sc.transform(Xn_va),     dtype=torch.float32)
        Xn_te_t = torch.tensor(sc.transform(X_num_te),  dtype=torch.float32)
        Xc_tr_t = torch.tensor(Xc_tr.values,    dtype=torch.long)
        Xc_va_t = torch.tensor(Xc_va.values,    dtype=torch.long)
        Xc_te_t = torch.tensor(X_cat_te.values, dtype=torch.long)
        y_tr_t  = torch.tensor(y_tr.values, dtype=torch.float32).view(-1, 1)
        y_va_t  = torch.tensor(y_va.values, dtype=torch.float32).view(-1, 1)

        torch.manual_seed(seed)
        loader = DataLoader(TensorDataset(Xn_tr_t, Xc_tr_t, y_tr_t),
                            batch_size=64, shuffle=True)
        model = HousePriceEmbedModel(Xn_tr_t.shape[1], cards, hidden=hidden,
                                     p=p, dim_mult=dim_mult)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        best, best_state, cnt = float("inf"), None, 0
        for _ in range(epochs):
            model.train()
            for xn, xc, yb in loader:
                opt.zero_grad()
                loss = train_crit(model(xn, xc), yb)      # ★ 학습은 지정된 손실
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                vl = EVAL(model(Xn_va_t, Xc_va_t), y_va_t).item()   # ★ 평가는 MSE
            if vl < best:
                best, best_state, cnt = vl, copy.deepcopy(model.state_dict()), 0
            else:
                cnt += 1
            if cnt >= patience:
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            oof[va_idx] = model(Xn_va_t, Xc_va_t).squeeze().numpy()
            test_preds.append(model(Xn_te_t, Xc_te_t))

    return (np.sqrt(((oof - y.values) ** 2).mean()),
            torch.stack(test_preds).mean(dim=0))


# ===== 비교 구성 =====
# delta 해석: 로그 공간 오차가 delta를 넘으면 선형 처리
#   전형적 오차가 약 0.13이므로 delta=0.5는 진짜 이상치만, 0.1은 상당수를 클립
configs = [
    (nn.MSELoss(),              "MSE (기준)"),
    (nn.HuberLoss(delta=0.5),   "Huber delta=0.5"),
    (nn.HuberLoss(delta=0.2),   "Huber delta=0.2"),
    (nn.HuberLoss(delta=0.1),   "Huber delta=0.1"),
]

print("=" * 58)
print("v28: 손실 함수 교체 (seed 42, fold_seed 42 고정)")
print("=" * 58)

results, best_score, best_test, best_label = [], None, None, None
for crit, label in configs:
    score, te = run(crit)
    results.append((label, score))
    print(f">>> {label:18s} | OOF {score:.4f}")
    if best_score is None or score < best_score:
        best_score, best_test, best_label = score, te, label

print("\n" + "=" * 58)
print(f"{'손실 함수':20s} | OOF RMSE")
print("-" * 58)
for label, score in results:
    print(f"{label:20s} | {score:.4f}")
print("=" * 58)
print(f"★ 최적: {best_label} → OOF {best_score:.4f}")
print("참조: MSE 단일(seed42) OOF 0.1298 / 임베딩 앙상블 Kaggle 0.126")

pred = torch.expm1(best_test).squeeze().numpy()
pd.DataFrame({"Id": test_ids, "SalePrice": pred}).to_csv("submission.csv", index=False)
print("submission.csv 저장 완료 (최적 손실 기준, 단일 시드)")