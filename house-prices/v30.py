# ==========================================================
#  v30_tabpfn.py — TabPFN 참조/앙상블 후보
#  사전학습 트랜스포머(in-context learning) — 별도 학습 없이 즉시 예측
#  ※ 최종 모델 사용 가능 여부는 교수님 확인 필요
#  프로토콜: MLP와 동일한 K-fold(random_state=42), 동일 이상치 처리
# ==========================================================

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor

from preprocess import HousePriceData

# ===== 데이터 (원-핫 표현, MLP와 동일 전처리) =====
X, y, X_test, test_ids, outlier_idx = HousePriceData(
    remove_outliers=True, fix_skew=True, full_ordinal=True
).prepare_full()
X = X.astype(float)
X_test = X_test.astype(float)

KF = KFold(n_splits=5, shuffle=True, random_state=42)     # v15~v29와 동일 분할
rmse = lambda p: np.sqrt(((p - y.values) ** 2).mean())

oof = np.zeros(len(X))
test_preds = []

print("=" * 58)
print("v30: TabPFN (사전학습 모델, 학습 불필요)")
print("=" * 58)

for fold, (tr_idx, va_idx) in enumerate(KF.split(X), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # 이상치는 fold-train에서만 제거 (MLP와 동일 조건)
    drop = outlier_idx.intersection(X_tr.index)
    X_tr, y_tr = X_tr.drop(index=drop), y_tr.drop(index=drop)

    # TabPFN은 내부적으로 정규화하므로 별도 스케일링 불필요
    model = TabPFNRegressor(device="cpu")                 # GPU 있으면 "cuda"
    model.fit(X_tr.values, y_tr.values)

    oof[va_idx] = model.predict(X_va.values)
    test_preds.append(model.predict(X_test.values))

    fold_rmse = np.sqrt(((oof[va_idx] - y_va.values) ** 2).mean())
    print(f"fold {fold} | train {len(X_tr)} / val {len(X_va)} | RMSE {fold_rmse:.4f}")

test_avg = np.mean(test_preds, axis=0)

print("\n" + "=" * 58)
print(f"★ TabPFN OOF : {rmse(oof):.4f}")
print("=" * 58)
print("참조: MLP 임베딩 앙상블 OOF 0.1266 / Kaggle 0.126")
print("      튜닝 트리(참조) OOF 0.1304")

# ===== 나중에 앙상블할 수 있도록 예측 저장 =====
np.save("tabpfn_oof.npy",  oof)
np.save("tabpfn_test.npy", test_avg)
print("\ntabpfn_oof.npy / tabpfn_test.npy 저장 (앙상블용)")

# ===== TabPFN 단독 제출 파일 =====
pred = np.expm1(test_avg)                                 # 로그 → 원래 가격
pd.DataFrame({"Id": test_ids, "SalePrice": pred}).to_csv("submission_tabpfn.csv", index=False)
print("submission_tabpfn.csv 저장 완료")