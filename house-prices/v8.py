import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader      # ★[v8] 미니배치용

from preprocess import HousePriceData
from model2 import HousePriceModelV2

torch.manual_seed(42)

# ===== 데이터 준비 (전체 262개 원-핫) =====
X_train, X_val, y_train, y_val, X_test, test_ids = HousePriceData().prepare()

# ===== ★[v8] 미니배치 구성 =====
train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)   # 64개씩, 매 epoch 섞음

# ===== 모델 / loss / optimizer =====
model = HousePriceModelV2(input_dim=X_train.shape[1], p=0.1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# ===== training / validation loop =====
epochs = 2000
best_val = float("inf")
patience = 100                                              # 미니배치는 수렴이 빨라 200→100
counter = 0

for epoch in range(epochs):
    # --- training: 미니배치 반복 ---
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:                            # ★[v8] 배치 단위로 학습
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(train_ds)                            # 에폭 평균 train loss

    # --- validation: 전체 한 번 (eval 모드) ---
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
        print(f"Epoch {epoch+1:4d} | train: {train_loss:.4f} | val: {val_loss:.4f} | val RMSE: {val_loss**0.5:.4f}")

    if counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1} | best val: {best_val:.4f} (RMSE {best_val**0.5:.4f})")
        break

model.load_state_dict(torch.load("best_model.pth"))

# ===== test 예측 → 제출 (이기면 주석 풀기) =====
model.eval()
with torch.no_grad():
    pred_log = model(X_test)
pred = torch.expm1(pred_log).squeeze().numpy()
submission = pd.DataFrame({"Id": test_ids, "SalePrice": pred})
submission.to_csv("submission.csv", index=False)
print("submission.csv 저장 완료:", submission.shape)