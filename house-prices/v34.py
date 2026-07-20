# ==========================================================
#  v34.py — FT-Transformer + MLP 임베딩 블렌드
#  가설: v20(구조 다양성)·v26(표현 블렌드) 실패는 모델들이 같은 계열이라
#        오차 구조가 유사했기 때문이다. FT(어텐션)와 MLP(완전연결)는 정보
#        처리 방식 자체가 달라 오차 상쇄 효과가 클 것이다.
#  ※ v26 교훈 — OOF로 가중치를 최적화하지 않고 고정값(0.6) 사용
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
CRIT = nn.MSELoss()
rmse = lambda p: np.sqrt(((p - y.values) ** 2).mean())


def run_mlp(seed, dim_mult=1, lr=0.002, hidden=(32, 8), p=0.1,
            epochs=3000, patience=200):
    """v25와 동일한 MLP 임베딩 구성"""
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
                loss = CRIT(model(xn, xc), yb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = CRIT(model(Xn_va_t, Xc_va_t), y_va_t).item()
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

    return oof, torch.stack(test_preds).mean(dim=0).squeeze().numpy()


# ===== FT 예측 불러오기 (v33에서 저장) =====
ft_oof  = np.load("ft_oof.npy")
ft_test = np.load("ft_test.npy")
print(f"FT 앙상블 OOF (불러옴) : {rmse(ft_oof):.4f}   [Kaggle 0.12473]")

# ===== MLP 앙상블 학습 =====
print("\nMLP 임베딩 앙상블 학습 중...")
seeds = [42, 0, 1, 2, 3]
mlp_oofs, mlp_tests = [], []
for s in seeds:
    o, t = run_mlp(s)
    mlp_oofs.append(o); mlp_tests.append(t)
    print(f"  seed {s:2d} | OOF {rmse(o):.4f}", flush=True)

mlp_oof  = np.mean(mlp_oofs, axis=0)
mlp_test = np.mean(mlp_tests, axis=0)
np.save("mlp_oof.npy", mlp_oof)
np.save("mlp_test.npy", mlp_test)
print(f"MLP 앙상블 OOF : {rmse(mlp_oof):.4f}   [Kaggle 0.126]")

# ===== 가중치별 OOF (참고용 — 선택에는 사용하지 않음) =====
print("\n가중치별 OOF (w = FT 비중) — 참고용")
for w in [0.0, 0.3, 0.5, 0.6, 0.7, 1.0]:
    print(f"  w={w:.1f} | OOF {rmse(w * ft_oof + (1 - w) * mlp_oof):.4f}")

# ===== 제출: w=0.6 고정 =====
# v26 교훈 — OOF로 가중치를 최적화하면 실제 성능과 어긋날 수 있음
# 성능이 더 좋은 FT에 소폭 가중하는 원칙적 선택
W = 0.6
blend_test = W * ft_test + (1 - W) * mlp_test
print(f"\n★ 제출 가중치 w={W} (FT {W:.0%} / MLP {1-W:.0%}) — 고정값")

pred = np.expm1(blend_test)
pd.DataFrame({"Id": test_ids, "SalePrice": pred}).to_csv("submission.csv", index=False)
print("submission.csv 저장 완료")