# ==========================================================
#  v36.py — 최종 블렌드 (가중치 FT 앙상블 + MLP 임베딩 앙상블)
#  재료:
#    ft_w_oof.npy / ft_w_test.npy  ← v35 (비정상 거래 가중치 적용 FT 앙상블)
#    mlp_oof.npy  / mlp_test.npy   ← v34 (MLP 임베딩 앙상블)
#  가중치: v34에서 검증된 w=0.6 (FT 60% / MLP 40%)
#          ※ v26 교훈 — OOF로 재최적화하지 않고 기존 검증값 사용
#  학습 없음 — 저장된 예측만 결합
# ==========================================================

import os
import numpy as np
import pandas as pd

W = 0.6                                    # FT 비중 (v34에서 검증)

# ===== 파일 확인 =====
need = ["ft_w_oof.npy", "ft_w_test.npy", "mlp_oof.npy", "mlp_test.npy"]
missing = [f for f in need if not os.path.exists(f)]
if missing:
    raise FileNotFoundError(
        f"다음 파일이 없습니다: {missing}\n"
        "  ft_w_*.npy → v35.py 실행 필요\n"
        "  mlp_*.npy  → v34.py 실행 필요"
    )

# ===== 예측 불러오기 =====
ft_oof,  ft_test  = np.load("ft_w_oof.npy"),  np.load("ft_w_test.npy")
mlp_oof, mlp_test = np.load("mlp_oof.npy"),   np.load("mlp_test.npy")

# ===== 정답 (OOF 평가용) =====
train = pd.read_csv("train.csv")
y = np.log1p(train["SalePrice"]).values
test_ids = pd.read_csv("test.csv")["Id"]

rmse = lambda p: np.sqrt(((p - y) ** 2).mean())

print("=" * 58)
print("v36: 최종 블렌드")
print("=" * 58)
print(f"FT  (가중치 적용) OOF : {rmse(ft_oof):.4f}")
print(f"MLP (임베딩)      OOF : {rmse(mlp_oof):.4f}")

# ===== 가중치별 OOF (참고용 — 선택에는 사용하지 않음) =====
print("\n가중치별 OOF (w = FT 비중) — 참고용")
for w in [0.0, 0.3, 0.5, 0.6, 0.7, 1.0]:
    mark = "  ★ 제출" if abs(w - W) < 1e-9 else ""
    print(f"  w={w:.1f} | OOF {rmse(w * ft_oof + (1 - w) * mlp_oof):.4f}{mark}")

# ===== 최종 결합 =====
blend_oof  = W * ft_oof  + (1 - W) * mlp_oof
blend_test = W * ft_test + (1 - W) * mlp_test

print("\n" + "=" * 58)
print(f"★ 최종 블렌드 OOF (w={W}) : {rmse(blend_oof):.4f}")
print("=" * 58)
print("참조 이력")
print("  v33 FT 앙상블(가중치 없음)     OOF 0.1226 / Kaggle 0.12473")
print("  v34 블렌드(가중치 없는 FT)     OOF 0.1207 / Kaggle 제출 대기")
print("  v25 MLP 앙상블                 OOF 0.1266 / Kaggle 0.126")

# ===== 제출 파일 =====
pred = np.expm1(blend_test)                 # 로그 공간 → 실제 가격
sub = pd.DataFrame({"Id": test_ids, "SalePrice": pred})
sub.to_csv("submission.csv", index=False)

print(f"\nsubmission.csv 저장 완료 ({sub.shape[0]}행)")
print(sub.head())
print(f"\n예측 가격 범위: ${pred.min():,.0f} ~ ${pred.max():,.0f} "
      f"(중앙값 ${np.median(pred):,.0f})")