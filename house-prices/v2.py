import torch
import torch.nn as nn
import pandas as pd

from preprocess import HousePriceData
from model import HousePriceModel

# ===== 데이터 준비 =====
torch.manual_seed(42) 
X_train, X_val, y_train, y_val, X_test, test_ids = HousePriceData(corr_threshold=0.7).prepare()

# ===== 모델 / loss / optimizer =====
model = HousePriceModel(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   # 규제(weight_decay) 없음 = v1

# ===== training / validation loop (+ early stopping / best checkpoint) =====
epochs = 1000
best_val = float("inf")
patience = 30
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
        torch.save(model.state_dict(), "best_model.pth")
        counter = 0
    else:
        counter += 1

    if (epoch + 1) % 20 == 0:
        # val RMSE = sqrt(val MSE) → baseline 0.37과 직접 비교 가능
        print(f"Epoch {epoch+1:4d} | train: {loss.item():.4f} | val: {val_loss:.4f} | val RMSE: {val_loss**0.5:.4f}")

    if counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1} | best val: {best_val:.4f} (RMSE {best_val**0.5:.4f})")
        break

# ===== 최고 모델 불러오기 =====
model.load_state_dict(torch.load("best_model.pth"))          # val 성능 최고였던 시점으로 복원

# ===== test 예측 → 제출 파일 생성 (실험 중엔 전체 주석) =====
model.eval()
with torch.no_grad():
    pred_log = model(X_test)                                 # 로그 공간 예측
pred = torch.expm1(pred_log).squeeze().numpy()               # 원래 가격으로 역변환
submission = pd.DataFrame({"Id": test_ids, "SalePrice": pred})
submission.to_csv("submission.csv", index=False)
print("submission.csv 저장 완료:", submission.shape)
print(submission.head())