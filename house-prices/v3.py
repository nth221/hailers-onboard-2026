import torch
import torch.nn as nn

from preprocess import HousePriceData
from model import HousePriceModel


# ★ 당신 코드(데이터준비 + 모델 + 학습루프)를 함수로 감쌌어요
def train_one(corr_threshold, epochs=1000, patience=30, lr=0.01):
    # ===== 데이터 준비 (★ 0.5 고정 → corr_threshold 인자로) =====
    torch.manual_seed(42)
    X_train, X_val, y_train, y_val, X_test, test_ids = \
        HousePriceData(corr_threshold=corr_threshold).prepare()

    # ===== 모델 / loss / optimizer (v2와 동일) =====
    model = HousePriceModel(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ===== training / validation loop (v2와 동일) =====
    best_val = float("inf")
    counter = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        if val_loss < best_val:
            best_val = val_loss
            counter = 0                      # ★ checkpoint 저장은 뺌(스윕엔 불필요)
        else:
            counter += 1
        if counter >= patience:
            break                            # ★ 매 epoch 출력도 뺌(5번 돌리면 너무 길어서)

    return X_train.shape[1], best_val ** 0.5  # ★ (변수 수, best val RMSE) 반환


# ===== ★ 여러 임계값 자동 실험 =====
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
results = []
for t in thresholds:
    n_feat, rmse = train_one(t)
    results.append((t, n_feat, rmse))
    print(f"threshold={t} | 변수 {n_feat}개 | best val RMSE {rmse:.4f}")

# ===== ★ 요약표 =====
print("\n===== 요약 =====")
print("threshold | 변수수 | val RMSE")
for t, n, r in results:
    print(f"   {t:>3}    |  {n:>3}  |  {r:.4f}")