import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor

from preprocess import HousePriceData

# ===== MLP와 완전히 동일한 데이터·분할 =====
X, y, X_test, test_ids, outlier_idx = \
    HousePriceData(remove_outliers=True).prepare_full()
X = X.astype(float)                                    # 트리는 스케일링 불필요

kf = KFold(n_splits=5, shuffle=True, random_state=42)  # ★ v15와 동일한 fold
oof = np.zeros(len(X))
fold_rmses = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # 이상치는 fold-train에서만 제거 (MLP와 동일 조건)
    drop = outlier_idx.intersection(X_tr.index)
    X_tr, y_tr = X_tr.drop(index=drop), y_tr.drop(index=drop)

    model = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, random_state=42
    )
    model.fit(X_tr, y_tr)

    pred = model.predict(X_va)
    oof[va_idx] = pred
    rmse = np.sqrt(((pred - y_va.values) ** 2).mean())
    fold_rmses.append(rmse)
    print(f"fold {fold} | val RMSE {rmse:.4f}")

oof_rmse = np.sqrt(((oof - y.values) ** 2).mean())
print("\n===== 참조 베이스라인 (최종 모델 아님) =====")
print(f"fold별 RMSE : {np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.4f}")
print(f"★ OOF RMSE  : {oof_rmse:.4f}")
print(f"  (MLP 참조: OOF 0.1338)")