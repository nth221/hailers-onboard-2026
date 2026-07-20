import torch
import torch.nn as nn
import pandas as pd

from preprocess import HousePriceData
from model2 import HousePriceModelV2                       # ★[v7] 새 모델(작게 + BatchNorm + Dropout)

torch.manual_seed(42)                                       # 재현성 고정

# ===== 데이터 준비 (전체 262개 원-핫) =====
X_train, X_val, y_train, y_val, X_test, test_ids = HousePriceData().prepare()

# ===== 모델 / loss / optimizer =====
model = HousePriceModelV2(input_dim=X_train.shape[1])       # ★[v7] 새 구조 사용
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)   # ★ 0.01 → 0.005 (BatchNorm 안정성)

# ===== training / validation loop (+ early stopping / best checkpoint) =====
epochs = 3000
best_val = float("inf")
patience = 200                                               # ★ 50 → 200 (노이즈에 성급히 안 멈추게)
counter = 0

for epoch in range(epochs):
    model.train()                                          # ★ BatchNorm/Dropout 학습 모드 (필수)
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()

    model.eval()                                           # ★ BatchNorm/Dropout 평가 모드 (필수)
    with torch.no_grad():
        val_loss = criterion(model(X_val), y_val).item()

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        counter = 0
    else:
        counter += 1

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:4d} | train: {loss.item():.4f} | val: {val_loss:.4f} | val RMSE: {val_loss**0.5:.4f}")

    if counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1} | best val: {best_val:.4f} (RMSE {best_val**0.5:.4f})")
        break

# ===== 최고 모델 불러오기 =====
model.load_state_dict(torch.load("best_model.pth"))

# ===== test 예측 → 제출 (이기면 주석 풀어서 제출) =====
#model.eval()
#with torch.no_grad():
#    pred_log = model(X_test)
#pred = torch.expm1(pred_log).squeeze().numpy()
#submission = pd.DataFrame({"Id": test_ids, "SalePrice": pred})
#submission.to_csv("submission.csv", index=False)
#print("submission.csv 저장 완료:", submission.shape)
#print(submission.head())