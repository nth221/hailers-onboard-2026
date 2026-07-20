import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor

from preprocess import HousePriceData


def cv_tree(fix_skew, full_ordinal):
    X, y, X_test, test_ids, outlier_idx = HousePriceData(
        remove_outliers=True, fix_skew=fix_skew, full_ordinal=full_ordinal
    ).prepare_full()
    X = X.astype(float)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)   # v15·v16과 동일 분할
    oof = np.zeros(len(X))
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        drop = outlier_idx.intersection(X_tr.index)
        X_tr, y_tr = X_tr.drop(index=drop), y_tr.drop(index=drop)

        m = HistGradientBoostingRegressor(               # v16 최적 설정 고정
            learning_rate=0.02, max_leaf_nodes=15,
            l2_regularization=1.0, max_iter=1000, random_state=42)
        m.fit(X_tr, y_tr)
        oof[va_idx] = m.predict(X_va)

    return np.sqrt(((oof - y.values) ** 2).mean()), X.shape[1]


configs = [
    (False, False, "기준 (둘 다 없음)"),
    (True,  False, "① 왜도 보정만"),
    (False, True,  "② 서열형 확장만"),
    (True,  True,  "①+② 둘 다"),
]

results = []
for fs, fo, label in configs:
    score, ncol = cv_tree(fs, fo)
    results.append((label, ncol, score))
    print(f"\n>>> {label:20s} | 변수 {ncol:3d}개 | OOF {score:.4f}")

print("\n" + "=" * 55)
print(f"{'조합':22s} | 변수  | OOF RMSE")
print("-" * 55)
for label, ncol, score in results:
    print(f"{label:22s} | {ncol:4d}  | {score:.4f}")
print("=" * 55)
print("참조: 튜닝 트리 0.1304 / MLP 0.1338")