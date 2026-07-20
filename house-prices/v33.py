# ==========================================================
#  v33.py — FT-Transformer 시드 앙상블
#  v31 결과: FT 단일 OOF 0.1274 (MLP 단일 0.1298, 트리 0.1304보다 우수,
#            MLP 5시드 앙상블 0.1266에 근접)
#  가설: v25에서 시드 앙상블은 Kaggle 실제 이득이 검증된 유일한 앙상블이다
#        (임베딩 MLP: 0.129 → 0.126). FT-Transformer 단일이 이미 MLP 앙상블에
#        근접하므로, 동일하게 시드 앙상블을 적용하면 OOF 0.124 근처에 도달해
#        Kaggle에서 현 챔피언(0.126)을 넘을 것이다.
#  고정: 왜도보정+서열형확장, d32×1블록, lr=0.005, batch 256, K-fold(rs=42)
#  변경: 모델 시드 5개(42, 0, 1, 2, 3)
# ==========================================================

import copy
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from preprocess import HousePriceData
from model_ft import FTTransformer

X_num, X_cat, cards, y, X_num_te, X_cat_te, test_ids, outlier_idx = HousePriceData(
    remove_outliers=True, fix_skew=True, full_ordinal=True
).prepare_embed()

KF = KFold(n_splits=5, shuffle=True, random_state=42)
CRIT = nn.MSELoss()
rmse = lambda p: np.sqrt(((p - y.values) ** 2).mean())


def run(d_token, n_blocks, lr, dropout=0.1, n_heads=4, seed=42,
        batch_size=256, epochs=400, patience=40, verbose=True):
    oof = np.zeros(len(X_num))
    test_preds = []
    n_params = None

    for fold, (tr_idx, va_idx) in enumerate(KF.split(X_num), 1):
        t_fold = time.time()

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

        torch.manual_seed(seed)
        loader = DataLoader(TensorDataset(Xn_tr_t, Xc_tr_t, y_tr_t),
                            batch_size=batch_size, shuffle=True)
        model = FTTransformer(Xn_tr_t.shape[1], cards, d_token=d_token,
                              n_blocks=n_blocks, n_heads=n_heads, dropout=dropout)
        n_params = model.n_params()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

        best, best_state, cnt, last_epoch = float("inf"), None, 0, 0
        for ep in range(epochs):
            model.train()
            for xn, xc, yb in loader:
                opt.zero_grad()
                loss = CRIT(model(xn, xc), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            with torch.no_grad():
                vl = CRIT(model(Xn_va_t, Xc_va_t), y_va_t).item()
            if vl < best:
                best, best_state, cnt = vl, copy.deepcopy(model.state_dict()), 0
            else:
                cnt += 1
            last_epoch = ep + 1
            if cnt >= patience:
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            oof[va_idx] = model(Xn_va_t, Xc_va_t).squeeze().numpy()
            test_preds.append(model(Xn_te_t, Xc_te_t))

        if verbose:
            print(f"    fold {fold}/5 | val RMSE {best**0.5:.4f} "
                  f"| epoch {last_epoch:3d} | {time.time()-t_fold:.0f}초", flush=True)

    return rmse(oof), torch.stack(test_preds).mean(dim=0), n_params, oof   # ★ oof 추가


# ===== 시드 앙상블 =====
D_TOKEN, N_BLOCKS, LR = 32, 1, 0.005          # v31 최적 구성
seeds = [42, 0, 1, 2, 3]

print("=" * 66)
print(f"v33: FT-Transformer 시드 앙상블 (d{D_TOKEN} × {N_BLOCKS}블록, lr={LR})")
print("=" * 66)

all_oof, all_test, singles = [], [], []
t_start = time.time()

for s in seeds:
    print(f"\n[seed {s}]", flush=True)
    score, te, npar, oof = run(D_TOKEN, N_BLOCKS, LR, seed=s)
    all_oof.append(oof)
    all_test.append(te)
    singles.append(score)
    print(f"  → seed {s} OOF {score:.4f}", flush=True)

# ===== 학습 실패 구성 제외 (v27에서 도입한 원칙적 기준) =====
singles_arr = np.array(singles)
median = np.median(singles_arr)
keep = singles_arr < median * 1.5
n_drop = int((~keep).sum())
if n_drop:
    print(f"\n⚠ 학습 실패로 제외: {n_drop}개 "
          f"(OOF {singles_arr[~keep].round(4).tolist()})")

kept_oof  = [o for o, k in zip(all_oof, keep) if k]
kept_test = [t for t, k in zip(all_test, keep) if k]

# ===== 앙상블 (로그 공간에서 평균) =====
oof_avg  = np.mean(kept_oof, axis=0)
test_avg = torch.stack(kept_test).mean(dim=0)

print("\n" + "=" * 66)
print(f"파라미터  : {npar:,d}")
print(f"사용 구성 : {len(kept_oof)}/{len(all_oof)}개  (총 {time.time()-t_start:.0f}초)")
print(f"개별 평균 : {singles_arr[keep].mean():.4f}")
print(f"개별 최고 : {singles_arr[keep].min():.4f}")
print(f"★ FT 앙상블 OOF : {rmse(oof_avg):.4f}")
print("=" * 66)
print("참조: FT 단일(seed42)     OOF 0.1274")
print("      MLP 5시드 앙상블    OOF 0.1266 / Kaggle 0.126  ← 현 챔피언")
print("      튜닝 트리(참조)     OOF 0.1304")
print("      ※ 앙상블 OOF는 선택 편향으로 부풀려짐 — 판정은 Kaggle로")

# ===== 저장 (나중에 MLP와 블렌드 실험용) =====
np.save("ft_oof.npy",  oof_avg)
np.save("ft_test.npy", test_avg.squeeze().numpy())

pred = torch.expm1(test_avg).squeeze().numpy()
pd.DataFrame({"Id": test_ids, "SalePrice": pred}).to_csv("submission.csv", index=False)
print("\nsubmission.csv / ft_oof.npy / ft_test.npy 저장 완료")