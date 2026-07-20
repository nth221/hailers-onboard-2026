# ==========================================================
#  v25.py — 임베딩 + 시드 앙상블
#  가설: v12에서 시드 앙상블은 Kaggle에서 실제 이득이 검증된 유일한 앙상블이다
#        (0.145 → 0.14067, −0.0043). 반면 v20의 구조 다양성 앙상블은 OOF만
#        개선되고 Kaggle은 무변화였다. 따라서 Kaggle로 검증된 임베딩 구성
#        (dim_mult=1, Kaggle 0.129)에 시드 앙상블을 적용하면 유사한 폭의
#        개선이 나타나 0.125 근처에 도달할 것이다.
#  고정: 왜도보정+서열형확장, dim_mult=1, 구조 32→8, lr=0.002, K-fold 분할
#  변경: 시드 5개(42, 0, 1, 2, 3)로 학습 후 예측 평균
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

# ===== 데이터 준비 (임베딩용: 범주형은 정수 인덱스) =====
X_num, X_cat, cards, y, X_num_te, X_cat_te, test_ids, outlier_idx = HousePriceData(
    remove_outliers=True, fix_skew=True, full_ordinal=True
).prepare_embed()

KF = KFold(n_splits=5, shuffle=True, random_state=42)    # v15~v24와 동일 분할


def run_seed(seed, dim_mult=1, lr=0.002, hidden=(32, 8), p=0.1,
             epochs=3000, patience=200):
    """한 시드로 5-fold 학습 → (OOF 예측, test 예측 평균)"""
    oof = np.zeros(len(X_num))
    test_preds = []

    for tr_idx, va_idx in KF.split(X_num):
        Xn_tr, Xn_va = X_num.iloc[tr_idx], X_num.iloc[va_idx]
        Xc_tr, Xc_va = X_cat.iloc[tr_idx], X_cat.iloc[va_idx]
        y_tr,  y_va  = y.iloc[tr_idx],     y.iloc[va_idx]

        # 이상치는 fold-train에서만 제거 (검증 fold는 손대지 않음)
        drop = outlier_idx.intersection(Xn_tr.index)
        Xn_tr = Xn_tr.drop(index=drop)
        Xc_tr = Xc_tr.drop(index=drop)
        y_tr  = y_tr.drop(index=drop)

        # 스케일링은 수치형만, fold-train에만 fit (누수 방지)
        sc = StandardScaler()
        Xn_tr_t = torch.tensor(sc.fit_transform(Xn_tr), dtype=torch.float32)
        Xn_va_t = torch.tensor(sc.transform(Xn_va),     dtype=torch.float32)
        Xn_te_t = torch.tensor(sc.transform(X_num_te),  dtype=torch.float32)
        Xc_tr_t = torch.tensor(Xc_tr.values,    dtype=torch.long)
        Xc_va_t = torch.tensor(Xc_va.values,    dtype=torch.long)
        Xc_te_t = torch.tensor(X_cat_te.values, dtype=torch.long)
        y_tr_t  = torch.tensor(y_tr.values, dtype=torch.float32).view(-1, 1)
        y_va_t  = torch.tensor(y_va.values, dtype=torch.float32).view(-1, 1)

        torch.manual_seed(seed)                              # ★ 시드만 다르게
        loader = DataLoader(TensorDataset(Xn_tr_t, Xc_tr_t, y_tr_t),
                            batch_size=64, shuffle=True)
        model = HousePriceEmbedModel(Xn_tr_t.shape[1], cards, hidden=hidden,
                                     p=p, dim_mult=dim_mult)
        crit = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=lr)

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

    return oof, torch.stack(test_preds).mean(dim=0)


# ===== 시드 5개로 학습 =====
seeds = [42, 0, 1, 2, 3]
all_oof, all_test, singles = [], [], []

print("=" * 62)
print("v25: 임베딩(dim_mult=1) + 시드 앙상블")
print("=" * 62)

for s in seeds:
    oof, te = run_seed(s)
    rmse = np.sqrt(((oof - y.values) ** 2).mean())
    all_oof.append(oof)
    all_test.append(te)
    singles.append(rmse)
    print(f"seed={s:3d} | OOF {rmse:.4f}")

# ===== 앙상블 (로그 공간에서 평균) =====
oof_avg  = np.mean(all_oof, axis=0)
test_avg = torch.stack(all_test).mean(dim=0)
ens_rmse = np.sqrt(((oof_avg - y.values) ** 2).mean())

print("\n" + "=" * 62)
print(f"개별 평균 : {np.mean(singles):.4f}")
print(f"개별 최고 : {min(singles):.4f}")
print(f"★ 앙상블  : {ens_rmse:.4f}")
print("=" * 62)
print("참조: 임베딩 단일 OOF 0.1298 / Kaggle 0.129")
print("      v12 시드 앙상블 실적: Kaggle 0.145 → 0.14067 (−0.0043)")

# ===== 제출 파일 =====
pred = torch.expm1(test_avg).squeeze().numpy()              # 로그 공간 평균 → 가격 복원
pd.DataFrame({"Id": test_ids, "SalePrice": pred}).to_csv("submission.csv", index=False)
print("submission.csv 저장 완료")