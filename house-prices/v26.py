# ==========================================================
#  v26.py — 표현 수준 앙상블 (원-핫 + 임베딩)
#  고정: 왜도보정+서열형확장, 이상치 제거, K-fold 분할(random_state=42)
#  구성: 원-핫 32→8 (5시드) + 임베딩 dim_mult=1 (5시드) → 가중 결합
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
from model_flex import HousePriceModelFlex as Flex
from model_embed import HousePriceEmbedModel

KF = KFold(n_splits=5, shuffle=True, random_state=42)     # 두 표현이 동일 분할 사용

# ===== 데이터 (두 표현 모두 준비) =====
Xo, y, Xo_te, test_ids, out_idx = HousePriceData(
    remove_outliers=True, fix_skew=True, full_ordinal=True).prepare_full()

Xn, Xc, cards, y2, Xn_te, Xc_te, _, out_idx2 = HousePriceData(
    remove_outliers=True, fix_skew=True, full_ordinal=True).prepare_embed()


def _train(model, loader, val_inputs, y_va_t, lr, epochs=3000, patience=200):
    """공통 학습 루프 → best state 복원된 모델 반환"""
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, cnt = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for batch in loader:
            opt.zero_grad()
            *xb, yb = batch
            loss = crit(model(*xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(*val_inputs), y_va_t).item()
        if vl < best:
            best, best_state, cnt = vl, copy.deepcopy(model.state_dict()), 0
        else:
            cnt += 1
        if cnt >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model


def run_onehot(seed, lr=0.002, hidden=(32, 8), p=0.1):
    oof, tests = np.zeros(len(Xo)), []
    for tr, va in KF.split(Xo):
        X_tr, X_va = Xo.iloc[tr], Xo.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        drop = out_idx.intersection(X_tr.index)
        X_tr, y_tr = X_tr.drop(index=drop), y_tr.drop(index=drop)

        sc = StandardScaler()
        X_tr_t = torch.tensor(sc.fit_transform(X_tr), dtype=torch.float32)
        X_va_t = torch.tensor(sc.transform(X_va),    dtype=torch.float32)
        X_te_t = torch.tensor(sc.transform(Xo_te),   dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32).view(-1, 1)
        y_va_t = torch.tensor(y_va.values, dtype=torch.float32).view(-1, 1)

        torch.manual_seed(seed)
        loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=True)
        model = Flex(input_dim=X_tr_t.shape[1], hidden=hidden, p=p)
        model = _train(model, loader, (X_va_t,), y_va_t, lr)

        with torch.no_grad():
            oof[va] = model(X_va_t).squeeze().numpy()
            tests.append(model(X_te_t))
    return oof, torch.stack(tests).mean(dim=0)


def run_embed(seed, lr=0.002, hidden=(32, 8), p=0.1, dim_mult=1):
    oof, tests = np.zeros(len(Xn)), []
    for tr, va in KF.split(Xn):
        Xn_tr, Xn_va = Xn.iloc[tr], Xn.iloc[va]
        Xc_tr, Xc_va = Xc.iloc[tr], Xc.iloc[va]
        y_tr, y_va   = y.iloc[tr],  y.iloc[va]
        drop = out_idx2.intersection(Xn_tr.index)
        Xn_tr, Xc_tr, y_tr = (Xn_tr.drop(index=drop),
                              Xc_tr.drop(index=drop), y_tr.drop(index=drop))

        sc = StandardScaler()
        Xn_tr_t = torch.tensor(sc.fit_transform(Xn_tr), dtype=torch.float32)
        Xn_va_t = torch.tensor(sc.transform(Xn_va),     dtype=torch.float32)
        Xn_te_t = torch.tensor(sc.transform(Xn_te),     dtype=torch.float32)
        Xc_tr_t = torch.tensor(Xc_tr.values, dtype=torch.long)
        Xc_va_t = torch.tensor(Xc_va.values, dtype=torch.long)
        Xc_te_t = torch.tensor(Xc_te.values, dtype=torch.long)
        y_tr_t  = torch.tensor(y_tr.values, dtype=torch.float32).view(-1, 1)
        y_va_t  = torch.tensor(y_va.values, dtype=torch.float32).view(-1, 1)

        torch.manual_seed(seed)
        loader = DataLoader(TensorDataset(Xn_tr_t, Xc_tr_t, y_tr_t),
                            batch_size=64, shuffle=True)
        model = HousePriceEmbedModel(Xn_tr_t.shape[1], cards, hidden=hidden,
                                     p=p, dim_mult=dim_mult)
        model = _train(model, loader, (Xn_va_t, Xc_va_t), y_va_t, lr)

        with torch.no_grad():
            oof[va] = model(Xn_va_t, Xc_va_t).squeeze().numpy()
            tests.append(model(Xn_te_t, Xc_te_t))
    return oof, torch.stack(tests).mean(dim=0)


seeds = [42, 0, 1, 2, 3]

print("=" * 62)
print("[1/2] 원-핫 표현 (5시드)")
o_oofs, o_tests = [], []
for s in seeds:
    oof, te = run_onehot(s)
    o_oofs.append(oof); o_tests.append(te)
    print(f"  seed={s:3d} | OOF {np.sqrt(((oof - y.values)**2).mean()):.4f}")

print("\n[2/2] 임베딩 표현 (5시드)")
e_oofs, e_tests = [], []
for s in seeds:
    oof, te = run_embed(s)
    e_oofs.append(oof); e_tests.append(te)
    print(f"  seed={s:3d} | OOF {np.sqrt(((oof - y.values)**2).mean()):.4f}")

# ===== 표현별 앙상블 =====
o_oof, o_test = np.mean(o_oofs, axis=0), torch.stack(o_tests).mean(dim=0)
e_oof, e_test = np.mean(e_oofs, axis=0), torch.stack(e_tests).mean(dim=0)

rmse = lambda p: np.sqrt(((p - y.values) ** 2).mean())
print("\n" + "=" * 62)
print(f"원-핫 앙상블 OOF : {rmse(o_oof):.4f}")
print(f"임베딩 앙상블 OOF : {rmse(e_oof):.4f}")

# ===== 가중 결합 스윕 (w = 임베딩 비중) =====
print("\n가중 결합 (w = 임베딩 비중)")
best_w, best_score = None, None
for w in [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]:
    score = rmse(w * e_oof + (1 - w) * o_oof)
    print(f"  w={w:.2f} | OOF {score:.4f}")
    if best_score is None or score < best_score:
        best_w, best_score = w, score

print("=" * 62)
print(f"★ 최적 w={best_w:.2f} → OOF {best_score:.4f}")
print("참조: 임베딩 단독 앙상블 OOF 0.1266 / Kaggle 0.126")

# ===== 제출 =====
blend = best_w * e_test + (1 - best_w) * o_test
pred = torch.expm1(blend).squeeze().numpy()
pd.DataFrame({"Id": test_ids, "SalePrice": pred}).to_csv("submission.csv", index=False)
print(f"submission.csv 저장 완료 (w={best_w:.2f})")