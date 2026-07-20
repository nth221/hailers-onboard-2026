# ==========================================================
#  v31.py — FT-Transformer (최소 구성 신호 확인용)
#  가설: 셀프 어텐션이 특성 간 상호작용을 데이터로부터 직접 학습하면,
#        v21의 수동 상호작용으로는 못 찾은 관계를 포착할 것이다.
#  고정: 왜도보정+서열형확장, K-fold(random_state=42), seed 42
#
#  ★속도 조정 (CPU 환경 고려)
#    - LayerNorm 사용 → 배치 크기 무관하므로 batch 256 (BatchNorm MLP와 다름)
#    - 블록 2→1 (어텐션 비용 절반), epochs 800→400, patience 80→40
#    - 구성 1개만 먼저 돌려 신호 확인 → 유망하면 확장
#    - fold별 진행 출력으로 소요 시간·성능을 즉시 확인
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


def run(d_token, n_blocks, lr, dropout=0.1, n_heads=4, seed=42,
        batch_size=256, epochs=400, patience=40):
    oof = np.zeros(len(X_num))
    test_preds = []
    n_params = None

    for fold, (tr_idx, va_idx) in enumerate(KF.split(X_num), 1):    # ★ fold 번호
        t_fold = time.time()                                         # ★ fold 시작 시각

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 트랜스포머 안정화
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

        # ★ fold별 진행 출력 (소요 시간·성능 즉시 확인)
        print(f"    fold {fold}/5 | val RMSE {best**0.5:.4f} "
              f"| epoch {last_epoch:3d} | {time.time()-t_fold:.0f}초", flush=True)

    return (np.sqrt(((oof - y.values) ** 2).mean()),
            torch.stack(test_preds).mean(dim=0), n_params)


# ===== 비교 구성 (먼저 1개만 — 유망하면 아래 주석 해제해 확장) =====
configs = [
    (32, 1, 0.005,  "d32 × 1블록, lr=0.005"),
    # (32, 2, 0.005,  "d32 × 2블록, lr=0.005"),
    # (16, 1, 0.005,  "d16 × 1블록, lr=0.005"),
]

print("=" * 66)
print("v31: FT-Transformer (batch 256, 1블록, epochs 400)")
print("=" * 66)

results, best_score, best_test, best_label = [], None, None, None
for d, nb, lr, label in configs:
    print(f"\n[{label}]", flush=True)
    t0 = time.time()
    score, te, npar = run(d, nb, lr)
    results.append((label, npar, score))
    print(f">>> {label:24s} | 파라미터 {npar:6,d} | OOF {score:.4f} "
          f"| 총 {time.time()-t0:.0f}초")
    if best_score is None or score < best_score:
        best_score, best_test, best_label = score, te, label

print("\n" + "=" * 66)
print(f"{'구성':26s} | 파라미터 | OOF RMSE")
print("-" * 66)
for label, npar, score in results:
    print(f"{label:26s} | {npar:7,d}  | {score:.4f}")
print("=" * 66)
print(f"★ 최적: {best_label} → OOF {best_score:.4f}")
print("참조: MLP 단일(seed42) OOF 0.1298 / MLP 앙상블 OOF 0.1266 (Kaggle 0.126)")
print("      튜닝 트리(참조) OOF 0.1304")

pred = torch.expm1(best_test).squeeze().numpy()
pd.DataFrame({"Id": test_ids, "SalePrice": pred}).to_csv("submission_ft.csv", index=False)
print("submission_ft.csv 저장 완료")