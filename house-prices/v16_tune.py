import itertools
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor

from preprocess import HousePriceData

X, y, X_test, test_ids, outlier_idx = \
    HousePriceData(remove_outliers=True).prepare_full()
X = X.astype(float)

kf = KFold(n_splits=5, shuffle=True, random_state=42)


def cv_score(**params):
    oof = np.zeros(len(X))
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        drop = outlier_idx.intersection(X_tr.index)
        X_tr, y_tr = X_tr.drop(index=drop), y_tr.drop(index=drop)
        m = HistGradientBoostingRegressor(random_state=42, **params)
        m.fit(X_tr, y_tr)
        oof[va_idx] = m.predict(X_va)
    return np.sqrt(((oof - y.values) ** 2).mean())


# 탐색 조합
lrs        = [0.02, 0.05]
leaves     = [15, 31]
l2s        = [0.0, 1.0]
iters      = [1000]

results = []
for lr, lf, l2, it in itertools.product(lrs, leaves, l2s, iters):
    score = cv_score(learning_rate=lr, max_leaf_nodes=lf,
                     l2_regularization=l2, max_iter=it)
    results.append((lr, lf, l2, score))
    print(f"lr={lr} leaves={lf:2d} l2={l2} → OOF {score:.4f}")

results.sort(key=lambda r: r[3])
print(f"\n★ 최적 참조 천장: OOF {results[0][3]:.4f} "
      f"(lr={results[0][0]}, leaves={results[0][1]}, l2={results[0][2]})")
print(f"  MLP 현재: 0.1338")