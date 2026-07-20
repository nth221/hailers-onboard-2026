# ==========================================================
#  v23.py — 임베딩 차원 탐색
#  가설: v22의 차원 상한 6은 임의 설정값이다. 차원이 과하면 파라미터 과잉으로
#        과적합, 부족하면 표현력 부족이 되므로 최적점이 존재할 것이다(v19와 동일 논리).
#  고정: 왜도보정+서열형확장, 구조 32→8, lr=0.002, K-fold 분할
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


def run(dim_cap, lr=0.002, hidden=(32, 8), p=0.1, seed=42, epochs=3000, patience=200):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X_num))
    test_preds = []
    in_dim = None

    for tr_idx, va_idx in kf.split(X_num):
        Xn_tr, Xn_va = X_num.iloc[tr_idx], X_num.iloc[va_idx]
        Xc_tr, Xc_va = X_cat.iloc[tr_idx], X_cat.iloc[va_idx]
        y_tr,  y_va  = y.iloc[tr_idx],     y.iloc[va_idx]

        drop = outlier_idx.intersection(Xn_tr.index)             # fold-train에서만 제거
        Xn_tr, Xc_tr, y_tr = (Xn_tr.drop(index=drop),
                              Xc_tr.drop(index=drop), y_tr.drop(index=drop))

        sc = StandardScaler()                                    # 수치형만 스케일링
        Xn_tr_t = torch.tensor(sc.fit_transform(Xn_tr), dtype=torch.float32)
        Xn_va_t = torch.tensor(sc.transform(Xn_va),     dtype=torch.float32)
        Xn_te_t = torch.tensor(sc.transform(X_num_te),  dtype=torch.float32)
        Xc_tr_t = torch.tensor(Xc_tr.values, dtype=torch.long)
        Xc_va_t = torch.tensor(Xc_va.values, dtype=torch.long)
        Xc_te_t = torch.tensor(X_cat_te.values, dtype=torch.long)
        y_tr_t  = torch.tensor(y_tr.values, dtype=torch.float32).view(-1, 1)
        y_va_t  = torch.tensor(y_va.values, dtype=torch.float32).view(-1, 1)

        torch.manual_seed(seed)
        loader = DataLoader(TensorDataset(Xn_tr_t, Xc_tr_t, y_tr_t),
                            batch_size=64, shuffle=True)
        model = HousePriceEmbedModel(Xn_tr_t.shape[1], cards,
                                     hidden=hidden, p=p, dim_cap=dim_cap)
        in_dim = model.input_dim
        crit, opt = nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=lr)

        best, best_state, cnt = float("inf"), None, 0
        for _ in range(epochs):
            model.train()
            for xn, xc, yb in loader:
                opt.zero_grad()
                loss = crit(model(xn, xc), yb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = crit(model(Xn_va_t, Xc_va_t), y_va_t).item()
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
            torch.stack(test_preds).mean(dim=0), in_dim)


print("=" * 60)
print("v23: 임베딩 차원 탐색 (구조 32→8, lr=0.002 고정)")
print("=" * 60)

results, best_score, best_test = [], None, None
for cap in [3, 4, 6, 8, 10]:
    score, te, in_dim = run(cap)
    results.append((cap, in_dim, score))
    print(f">>> dim_cap={cap:2d} | 입력차원 {in_dim:3d} | OOF {score:.4f}")
    if best_score is None or score < best_score:
        best_score, best_test = score, te

print("\n" + "=" * 60)
print(f"{'dim_cap':>8s} | 입력차원 | OOF RMSE")
print("-" * 60)
for cap, in_dim, score in results:
    print(f"{cap:8d} | {in_dim:7d}  | {score:.4f}")
print("=" * 60)
print("참조: 원-핫 224차원 OOF 0.1278 / Kaggle 0.13343 → 임베딩 Kaggle 0.129")

pred = torch.expm1(best_test).squeeze().numpy()
pd.DataFrame({"Id": test_ids, "SalePrice": pred}).to_csv("submission.csv", index=False)
print("submission.csv 저장 완료 (최적 dim_cap 기준)")