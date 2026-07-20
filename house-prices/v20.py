import copy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from preprocess import HousePriceData
from model_flex import HousePriceModelFlex as Flex

# ===== 데이터 (현재 최적 표현) =====
X, y, X_test_raw, test_ids, outlier_idx = HousePriceData(
    remove_outliers=True, fix_skew=True, full_ordinal=True
).prepare_full()

kf = KFold(n_splits=5, shuffle=True, random_state=42)


def run_config(hidden, seed, lr=0.002, p=0.1):
    """한 구성(구조+시드)으로 5-fold 학습 → (OOF 예측, test 예측 평균)"""
    oof = np.zeros(len(X))
    test_preds = []

    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        drop = outlier_idx.intersection(X_tr.index)
        X_tr, y_tr = X_tr.drop(index=drop), y_tr.drop(index=drop)

        sc = StandardScaler()
        X_tr_t = torch.tensor(sc.fit_transform(X_tr), dtype=torch.float32)
        X_va_t = torch.tensor(sc.transform(X_va),      dtype=torch.float32)
        X_te_t = torch.tensor(sc.transform(X_test_raw), dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32).view(-1, 1)
        y_va_t = torch.tensor(y_va.values, dtype=torch.float32).view(-1, 1)

        torch.manual_seed(seed)
        loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=True)
        model = Flex(input_dim=X_tr_t.shape[1], hidden=hidden, p=p)
        crit, opt = nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=lr)

        best, best_state, cnt = float("inf"), None, 0
        for _ in range(3000):
            model.train()
            for xb, yb in loader:
                opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vl = crit(model(X_va_t), y_va_t).item()
            if vl < best:
                best, best_state, cnt = vl, copy.deepcopy(model.state_dict()), 0
            else:
                cnt += 1
            if cnt >= 200:
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            oof[va_idx] = model(X_va_t).squeeze().numpy()
            test_preds.append(model(X_te_t))

    return oof, torch.stack(test_preds).mean(dim=0)


# ===== 구조 × 시드 조합 =====
configs = [
    ((32, 8),   42), ((32, 8),   0),
    ((64, 16),  42), ((64, 16),  0),
    ((128, 32), 42), ((128, 32), 0),
]

all_oof, all_test = [], []
for hidden, seed in configs:
    oof, te = run_config(hidden, seed)
    rmse = np.sqrt(((oof - y.values) ** 2).mean())
    all_oof.append(oof); all_test.append(te)
    print(f"{str(hidden):10s} seed={seed:2d} | OOF {rmse:.4f}")

# ===== 앙상블 =====
oof_avg  = np.mean(all_oof, axis=0)
test_avg = torch.stack(all_test).mean(dim=0)
ens_rmse = np.sqrt(((oof_avg - y.values) ** 2).mean())

print("\n===== 결과 =====")
print(f"개별 평균 : {np.mean([np.sqrt(((o - y.values)**2).mean()) for o in all_oof]):.4f}")
print(f"개별 최고 : {min(np.sqrt(((o - y.values)**2).mean()) for o in all_oof):.4f}")
print(f"★ 앙상블  : {ens_rmse:.4f}   (단일 최고 0.1278)")

pred = torch.expm1(test_avg).squeeze().numpy()
pd.DataFrame({"Id": test_ids, "SalePrice": pred}).to_csv("submission.csv", index=False)
print("submission.csv 저장 완료")