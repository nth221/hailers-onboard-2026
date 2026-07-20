# ==========================================================
#  v29.py — 코사인 annealing + 스냅샷 앙상블
#  v10 실패 원인: ReduceLROnPlateau가 val 노이즈를 오판해 lr 붕괴
#               → 코사인은 val을 안 보므로 그 실패 모드가 제거됨
#  보너스: warm restart 각 주기 끝의 모델을 모아 스냅샷 앙상블 구성
#         (한 번 학습으로 앙상블 멤버 확보 = v27보다 효율적)
#  고정: 임베딩 dim_mult=1, 구조 32→8, MSE, seed 42, fold_seed 42
# ==========================================================

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


def run_snapshot(lr=0.002, cycle_len=200, n_cycles=5, eta_min=1e-5,
                 seed=42, dim_mult=1, hidden=(32, 8), p=0.1):
    """코사인 warm restart로 학습하며 각 주기 끝에 스냅샷 예측을 수집"""
    # snap_oof[k] = k번째 스냅샷의 전체 OOF 예측
    snap_oof = [np.zeros(len(X_num)) for _ in range(n_cycles)]
    snap_test = [[] for _ in range(n_cycles)]

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

        torch.manual_seed(seed)
        loader = DataLoader(TensorDataset(Xn_tr_t, Xc_tr_t, y_tr_t),
                            batch_size=64, shuffle=True)
        model = HousePriceEmbedModel(Xn_tr_t.shape[1], cards, hidden=hidden,
                                     p=p, dim_mult=dim_mult)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        # ★ 코사인 warm restart: T_0 주기마다 lr이 lr→eta_min으로 내려갔다 재시작
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=cycle_len, T_mult=1, eta_min=eta_min)

        k = 0                                              # 스냅샷 인덱스
        for epoch in range(cycle_len * n_cycles):
            model.train()
            for xn, xc, yb in loader:
                opt.zero_grad()
                loss = CRIT(model(xn, xc), yb)
                loss.backward()
                opt.step()
            sched.step()                                   # ★ epoch 단위로 스케줄 진행

            # ★ 주기 끝(lr 최저점) = 수렴 지점 → 스냅샷 저장
            if (epoch + 1) % cycle_len == 0:
                model.eval()
                with torch.no_grad():
                    snap_oof[k][va_idx] = model(Xn_va_t, Xc_va_t).squeeze().numpy()
                    snap_test[k].append(model(Xn_te_t, Xc_te_t))
                k += 1

    # fold별 test 예측을 스냅샷마다 평균
    snap_test = [torch.stack(t).mean(dim=0) for t in snap_test]
    return snap_oof, snap_test


rmse = lambda p: np.sqrt(((p - y.values) ** 2).mean())

print("=" * 60)
print("v29: 코사인 annealing + 스냅샷 앙상블 (seed 42, fold_seed 42)")
print("=" * 60)

snap_oof, snap_test = run_snapshot()

# ===== 스냅샷별 성능 =====
print("\n스냅샷별 OOF (주기 끝마다 저장)")
for k, o in enumerate(snap_oof, 1):
    print(f"  스냅샷 {k} (epoch {200*k:4d}) | OOF {rmse(o):.4f}")

# ===== 두 가지 결과 비교 =====
last_oof,  last_test  = snap_oof[-1], snap_test[-1]                 # 스케줄 효과만
ens_oof = np.mean(snap_oof, axis=0)                                 # 스케줄 + 스냅샷 앙상블
ens_test = torch.stack(snap_test).mean(dim=0)

print("\n" + "=" * 60)
print(f"마지막 스냅샷 단독 : {rmse(last_oof):.4f}   ← 코사인 스케줄 효과")
print(f"★ 스냅샷 앙상블    : {rmse(ens_oof):.4f}   ← 스케줄 + 앙상블 효과")
print("=" * 60)
print("참조: 기존 방식(고정 lr + early stopping, seed 42) OOF 0.1298")
print("      임베딩 5시드 앙상블 Kaggle 0.126")

# ===== 제출 (스냅샷 앙상블 기준) =====
pred = torch.expm1(ens_test).squeeze().numpy()
pd.DataFrame({"Id": test_ids, "SalePrice": pred}).to_csv("submission.csv", index=False)
print("submission.csv 저장 완료 (스냅샷 앙상블)")