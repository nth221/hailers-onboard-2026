# ==========================================================
#  v35.py — FT-Transformer 앙상블 + 비정상 거래 가중치
#  진단(v32): 오차 상위 60채에 Abnorml(3.9배)·Family(4.8배)가 과다 대표.
#             이들은 압류·급매·가족거래로 시장가가 아닌 가격에 거래됨.
#  검증(트리): 가중치 1.0→0.5에서 OOF 0.1307→0.1294, 최저가구간 0.1786→0.1749
#  가설: 비정상 거래의 학습 가중치를 낮추면 모델이 시장가 관계를 더 정확히
#        학습해, 다수(82%)를 차지하는 Normal 거래 예측이 개선될 것이다.
#  ※ 학습에만 가중치 적용, 검증·평가는 동등 가중 (v13 원칙)
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

# ===== ★[v35] 비정상 거래 가중치 =====
# prepare_embed()는 SaleCondition을 원본 형태로 반환하지 않으므로 별도로 읽음
W_ABN = 0.5                                          # 트리 검증에서 0.5가 최적
_raw = pd.read_csv("train.csv")
sample_w = pd.Series(
    np.where(_raw["SaleCondition"].isin(["Abnorml", "Family"]), W_ABN, 1.0),
    index=_raw.index
)
print(f"[가중치] 비정상 거래 {int((sample_w < 1).sum())}채에 w={W_ABN} 적용 "
      f"(전체 {len(sample_w)}채)")

KF = KFold(n_splits=5, shuffle=True, random_state=42)
CRIT = nn.MSELoss(reduction="none")                  # ★[v35] 샘플별 손실
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

        # ★[v35] 가중치를 이상치 제거 후의 인덱스에 맞춰 정렬
        w_tr = sample_w.loc[Xn_tr.index].values

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
        w_tr_t  = torch.tensor(w_tr, dtype=torch.float32).view(-1, 1)   # ★[v35]

        torch.manual_seed(seed)
        loader = DataLoader(TensorDataset(Xn_tr_t, Xc_tr_t, y_tr_t, w_tr_t),  # ★ 가중치 포함
                            batch_size=batch_size, shuffle=True)
        model = FTTransformer(Xn_tr_t.shape[1], cards, d_token=d_token,
                              n_blocks=n_blocks, n_heads=n_heads, dropout=dropout)
        n_params = model.n_params()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

        best, best_state, cnt, last_epoch = float("inf"), None, 0, 0
        for ep in range(epochs):
            model.train()
            for xn, xc, yb, wb in loader:                        # ★ wb 추가
                opt.zero_grad()
                loss = (CRIT(model(xn, xc), yb) * wb).mean()     # ★ 가중 평균
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            with torch.no_grad():
                # ★ 검증은 가중치 없이 (평가는 모든 샘플 동등)
                vl = CRIT(model(Xn_va_t, Xc_va_t), y_va_t).mean().item()
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

    return rmse(oof), torch.stack(test_preds).mean(dim=0), n_params, oof


# ===== 시드 앙상블 =====
D_TOKEN, N_BLOCKS, LR = 32, 1, 0.005
seeds = [42, 0, 1, 2, 3]        # 다시 5개로

print("=" * 66)
print(f"v35: FT 앙상블 + 비정상 거래 가중치 (w={W_ABN})")
print("=" * 66)

all_oof, all_test, singles = [], [], []
t_start = time.time()

for s in seeds:
    print(f"\n[seed {s}]", flush=True)
    score, te, npar, oof = run(D_TOKEN, N_BLOCKS, LR, seed=s)
    all_oof.append(oof); all_test.append(te); singles.append(score)
    print(f"  → seed {s} OOF {score:.4f}", flush=True)

# ===== 학습 실패 구성 제외 (v27 기준) =====
sa = np.array(singles)
keep = sa < np.median(sa) * 1.5
if (~keep).sum():
    print(f"\n⚠ 학습 실패 제외: {(~keep).sum()}개 (OOF {sa[~keep].round(4).tolist()})")

kept_oof  = [o for o, k in zip(all_oof, keep) if k]
kept_test = [t for t, k in zip(all_test, keep) if k]

oof_avg  = np.mean(kept_oof, axis=0)
test_avg = torch.stack(kept_test).mean(dim=0)

print("\n" + "=" * 66)
print(f"사용 구성 : {len(kept_oof)}/{len(all_oof)}개  (총 {time.time()-t_start:.0f}초)")
print(f"개별 평균 : {sa[keep].mean():.4f}")
print(f"개별 최고 : {sa[keep].min():.4f}")
print(f"★ 앙상블 OOF : {rmse(oof_avg):.4f}")
print("=" * 66)
print("참조: v33 FT 앙상블(가중치 없음) OOF 0.1226 / Kaggle 0.12473  ← 현 챔피언")
print("      트리 검증: 가중치 적용 시 OOF −0.0013")

np.save("ft_w_oof.npy",  oof_avg)
np.save("ft_w_test.npy", test_avg.squeeze().numpy())

pred = torch.expm1(test_avg).squeeze().numpy()
pd.DataFrame({"Id": test_ids, "SalePrice": pred}).to_csv("submission.csv", index=False)
print("\nsubmission.csv / ft_w_oof.npy / ft_w_test.npy 저장 완료")