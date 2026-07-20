# ==========================================================
#  v27.py — 임베딩 앙상블 확장 (모델 시드 × fold 분할 시드)
#  v26 교훈: 표현이 다른 모델 블렌드는 OOF↔Kaggle 간격 차이로 실패
#           → 검증된 임베딩 구성만 유지하고 평균 대상만 확대
#  고정: 왜도보정+서열형확장, dim_mult=1, 구조 32→8, lr=0.002
#  변경: 모델 시드 5→8개, fold 분할 시드 1→2개 (총 8×2×5 = 80개 모델)
#  ★추가: 학습 실패 구성 자동 제외 (OOF > 중앙값×1.5)
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


def run_one(model_seed, fold_seed, dim_mult=1, lr=0.002,
            hidden=(32, 8), p=0.1, epochs=3000, patience=200):
    """모델 시드 + fold 분할 시드 한 쌍으로 5-fold 학습"""
    kf = KFold(n_splits=5, shuffle=True, random_state=fold_seed)   # ★ 분할 시드
    oof = np.zeros(len(X_num))
    test_preds = []

    for tr_idx, va_idx in kf.split(X_num):
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
        Xc_tr_t = torch.tensor(Xc_tr.values,    dtype=torch.long)   # 임베딩 입력은 정수
        Xc_va_t = torch.tensor(Xc_va.values,    dtype=torch.long)
        Xc_te_t = torch.tensor(X_cat_te.values, dtype=torch.long)
        y_tr_t  = torch.tensor(y_tr.values, dtype=torch.float32).view(-1, 1)
        y_va_t  = torch.tensor(y_va.values, dtype=torch.float32).view(-1, 1)

        torch.manual_seed(model_seed)                              # ★ 모델 시드
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


# ===== 실행 설정 (시간이 오래 걸리면 이 두 리스트를 줄이세요) =====
model_seeds = [42, 0, 1, 2, 3, 7]     # 8 → 6개
fold_seeds  = [42, 7]                  # 그대로

rmse = lambda p: np.sqrt(((p - y.values) ** 2).mean())
total = len(model_seeds) * len(fold_seeds)

print("=" * 64)
print(f"v27: 임베딩 앙상블 확장 — 모델시드 {len(model_seeds)} × 분할시드 "
      f"{len(fold_seeds)} = {total}개 구성 ({total * 5}개 모델)")
print("=" * 64)

all_oof, all_test, labels = [], [], []
i = 0
for fs in fold_seeds:
    fs_oof = []
    for ms in model_seeds:
        i += 1
        oof, te = run_one(ms, fs)
        all_oof.append(oof)
        all_test.append(te)
        labels.append(f"fold{fs}/model{ms}")
        fs_oof.append(oof)
        print(f"[{i:2d}/{total}] fold_seed={fs:3d} model_seed={ms:3d} "
              f"| OOF {rmse(oof):.4f}")
    print(f"  └ fold_seed={fs} 소계 앙상블 OOF: {rmse(np.mean(fs_oof, axis=0)):.4f}\n")

# ===== ★ 학습 실패 구성 제외 =====
# 기준: OOF가 중앙값의 1.5배를 넘으면 학습 실패로 간주 (발산·조기중단 등)
# 체리피킹이 아닌 사전 정의된 원칙적 기준
singles = np.array([rmse(o) for o in all_oof])
median = np.median(singles)
threshold = median * 1.5
keep = singles < threshold

n_drop = int((~keep).sum())
if n_drop:
    dropped = [f"{labels[j]}({singles[j]:.4f})" for j in range(len(singles)) if not keep[j]]
    print(f"⚠ 학습 실패로 제외: {n_drop}개 → {', '.join(dropped)}")
    print(f"  (중앙값 {median:.4f} × 1.5 = 기준 {threshold:.4f})\n")

kept_oof  = [o for o, k in zip(all_oof, keep) if k]
kept_test = [t for t, k in zip(all_test, keep) if k]

# ===== 최종 앙상블 =====
oof_avg  = np.mean(kept_oof, axis=0)
test_avg = torch.stack(kept_test).mean(dim=0)

print("=" * 64)
print(f"사용 구성 : {len(kept_oof)}/{len(all_oof)}개")
print(f"개별 평균 : {singles[keep].mean():.4f}")
print(f"개별 최고 : {singles[keep].min():.4f}")
print(f"★ 전체 앙상블 OOF : {rmse(oof_avg):.4f}")
print("=" * 64)
print("참조: v25 임베딩 앙상블(5시드) OOF 0.1266 / Kaggle 0.126")
print("      ※ 앙상블 OOF는 선택 편향으로 부풀려짐 — 판정은 Kaggle로")

# ===== 제출 =====
pred = torch.expm1(test_avg).squeeze().numpy()              # 로그 공간 평균 → 가격 복원
pd.DataFrame({"Id": test_ids, "SalePrice": pred}).to_csv("submission.csv", index=False)
print("submission.csv 저장 완료")