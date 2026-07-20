# ==========================================================
#  v37_tabfm.py — TabFM (Google 사전학습 표 데이터 모델)
#  in-context learning — 파라미터 학습 없이 즉시 예측
#  프로토콜: 우리 K-fold(random_state=42)·이상치 처리와 동일
#  ※ TabFM은 혼합 타입 DataFrame을 직접 처리하므로 원-핫 미적용
# ==========================================================

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from tabfm import TabFMRegressor
from tabfm import tabfm_v1_0_0_jax as tabfm_v1_0_0

# ===== 데이터 준비 (결측 처리만, 인코딩은 TabFM에 위임) =====
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
test_ids = test["Id"]

mask = (train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)
outlier_idx = train.index[mask]
print(f"[이상치 탐지] {len(outlier_idx)}건 (fold-train에서만 제거)")

y = np.log1p(train["SalePrice"])
train = train.drop(columns=["SalePrice"])
n = len(train)
full = pd.concat([train, test], axis=0)

# 결측 처리 (우리 파이프라인과 동일)
none_cols = ["PoolQC","MiscFeature","Alley","Fence","MasVnrType","FireplaceQu",
             "GarageType","GarageFinish","GarageQual","GarageCond",
             "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2"]
full[none_cols] = full[none_cols].fillna("None")
full[["MasVnrArea","GarageYrBlt"]] = full[["MasVnrArea","GarageYrBlt"]].fillna(0)
full["LotFrontage"] = full.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
for c in full.columns:
    full[c] = full[c].fillna(full[c].mode()[0]) if full[c].dtype == "object" else full[c].fillna(0)

full = full.drop(columns=["Id"])
X      = full.iloc[:n].reset_index(drop=True)
X_test = full.iloc[n:].reset_index(drop=True)
print(f"[준비 완료] 학습 {len(X)}개 / 변수 {X.shape[1]}개 "
      f"(범주형 {X.select_dtypes(include='object').shape[1]}개는 원본 유지)")

# ===== 모델 로드 (가중치 다운로드 — 최초 1회) =====
print("\nTabFM 가중치 로드 중...")
model = tabfm_v1_0_0.load(model_type="regression")

KF = KFold(n_splits=5, shuffle=True, random_state=42)
rmse = lambda p: np.sqrt(((p - y.values) ** 2).mean())

oof = np.zeros(len(X))
test_preds = []

print("=" * 56)
print("v37: TabFM")
print("=" * 56)

t_start = time.time()
for fold, (tr_idx, va_idx) in enumerate(KF.split(X), 1):
    t0 = time.time()
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # 이상치는 fold-train에서만 제거
    drop = outlier_idx.intersection(X_tr.index)
    X_tr, y_tr = X_tr.drop(index=drop), y_tr.drop(index=drop)

    reg = TabFMRegressor(model=model)
    reg.fit(X_tr, y_tr.values)                    # 학습이 아니라 문맥 등록

    oof[va_idx] = reg.predict(X_va)
    test_preds.append(reg.predict(X_test))

    fr = np.sqrt(((oof[va_idx] - y_va.values) ** 2).mean())
    print(f"fold {fold} | RMSE {fr:.4f} | {time.time()-t0:.0f}초", flush=True)

test_avg = np.mean(test_preds, axis=0)

print("\n" + "=" * 56)
print(f"★ TabFM OOF : {rmse(oof):.4f}   (총 {time.time()-t_start:.0f}초)")
print("=" * 56)
print("참조: FT 앙상블      OOF 0.1226 / Kaggle 0.12473")
print("      FT+MLP 블렌드  OOF 0.1207 / Kaggle 0.122   ← 현 챔피언")
print("      튜닝 트리(참조) OOF 0.1304")

# ===== 저장 (블렌드용) =====
np.save("tabfm_oof.npy",  oof)
np.save("tabfm_test.npy", test_avg)

pred = np.expm1(test_avg)
pd.DataFrame({"Id": test_ids, "SalePrice": pred}).to_csv("submission_tabfm.csv", index=False)
print("\nsubmission_tabfm.csv / tabfm_oof.npy / tabfm_test.npy 저장 완료")