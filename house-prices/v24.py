# ==========================================================
#  v24.py — 임베딩 차원 확대 탐색
#  v23 결과: cap 6/8/10이 동일 결과(0.1298) — 공식상 최대 차원이 5라 cap이
#            작동하지 않았음. 실제로는 "더 작게"(3,4)만 테스트된 셈이며 둘 다 악화.
#  가설: 현재 임베딩이 과도하게 압축돼(범주형 22개가 평균 2.4차원) 표현력이
#        부족할 수 있다. 배수를 키워 차원을 늘리면 OOF가 0.1298보다 낮아질 것이다.
#  고정: 왜도보정+서열형확장, 구조 32→8, lr=0.002, K-fold 분할
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


def run(dim_mult, lr=0.002, hidden=(32, 8), p=0.1, seed=42,
        epochs=3000, patience=200):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)        # v15~v23과 동일 분할
    oof = np.zeros(len(X_num))
    test_preds = []
    in_dim, emb_total = None, None

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
        Xn_tr_t = torch.tensor(sc.fit_transform(Xn_tr),  dtype=torch.float32)
        Xn_va_t = torch.tensor(sc.transform(Xn_va),      dtype=torch.float32)
        Xn_te_t = torch.tensor(sc.transform(X_num_te),   dtype=torch.float32)
        Xc_tr_t = torch.tensor(Xc_tr.values,    dtype=torch.long)   # 임베딩 입력은 정수
        Xc_va_t = torch.tensor(Xc_va.values,    dtype=torch.long)
        Xc_te_t = torch.tensor(X_cat_te.values, dtype=torch.long)
        y_tr_t  = torch.tensor(y_tr.values, dtype=torch.float32).view(-1, 1)
        y_va_t  = torch.tensor(y_va.values, dtype=torch.float32).view(-1, 1)

        torch.manual_seed(seed)
        loader = DataLoader(TensorDataset(Xn_tr_t, Xc_tr_t, y_tr_t),
                            batch_size=64, shuffle=True)
        model = HousePriceEmbedModel(Xn_tr_t.shape[1], cards, hidden=hidden,
                                     p=p, dim_mult=dim_mult)         # ★[v24]
        in_dim, emb_total = model.input_dim, model.emb_total
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

    return (np.sqrt(((oof - y.values) ** 2).mean()),
            torch.stack(test_preds).mean(dim=0), in_dim, emb_total)


print("=" * 66)
print("v24: 임베딩 차원 확대 탐색 (구조 32→8, lr=0.002 고정)")
print("=" * 66)

results, best_score, best_test, best_mult = [], None, None, None
for mult in [1, 2, 3]:
    score, te, in_dim, emb_total = run(mult)
    results.append((mult, emb_total, in_dim, score))
    print(f">>> dim_mult={mult} | 임베딩 {emb_total:3d}차원 "
          f"| 입력 {in_dim:3d} | OOF {score:.4f}")
    if best_score is None or score < best_score:
        best_score, best_test, best_mult = score, te, mult

print("\n" + "=" * 66)
print(f"{'dim_mult':>9s} | 임베딩차원 | 입력차원 | OOF RMSE")
print("-" * 66)
for mult, emb_total, in_dim, score in results:
    print(f"{mult:9d} | {emb_total:9d}  | {in_dim:7d}  | {score:.4f}")
print("=" * 66)
print(f"★ 최적 dim_mult={best_mult} → OOF {best_score:.4f}")
print("참조: 임베딩 mult=1 OOF 0.1298 / Kaggle 0.129")
print("      원-핫 224차원 OOF 0.1278 / Kaggle 0.13343")

# ===== 최적 구성으로 제출 파일 생성 =====
pred = torch.expm1(best_test).squeeze().numpy()
pd.DataFrame({"Id": test_ids, "SalePrice": pred}).to_csv("submission.csv", index=False)
print("submission.csv 저장 완료 (최적 dim_mult 기준)")