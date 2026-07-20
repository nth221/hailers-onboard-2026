import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from preprocess import HousePriceData
from model2 import HousePriceModelV2

torch.manual_seed(42)

X_train, X_val, y_train, y_val, X_test, test_ids = HousePriceData().prepare()

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

model = HousePriceModelV2(input_dim=X_train.shape[1], p=0.1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# ★[v10] 학습률 스케줄러: val이 20 epoch 정체하면 lr을 절반으로
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5,
    patience=100,          # ★ 20 → 100 (노이즈에 성급히 반응 안 하게)
    min_lr=1e-4            # ★ lr 하한선 (0으로 붕괴 방지)
)

epochs = 3000
best_val = float("inf")
patience = 150
counter = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(X_train)

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val), y_val).item()

    scheduler.step(val_loss)                               # ★[v10] val 기준으로 lr 조정

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        counter = 0
    else:
        counter += 1

    if (epoch + 1) % 20 == 0:
        cur_lr = optimizer.param_groups[0]["lr"]           # ★[v10] 현재 lr 확인용
        print(f"Epoch {epoch+1:4d} | train: {train_loss:.4f} | val: {val_loss:.4f} | "
              f"val RMSE: {val_loss**0.5:.4f} | lr: {cur_lr:.6f}")

    if counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1} | best val: {best_val:.4f} (RMSE {best_val**0.5:.4f})")
        break

model.load_state_dict(torch.load("best_model.pth"))

model.eval()
with torch.no_grad():
    pred_log = model(X_test)
pred = torch.expm1(pred_log).squeeze().numpy()
submission = pd.DataFrame({"Id": test_ids, "SalePrice": pred})
submission.to_csv("submission.csv", index=False)
print("submission.csv 저장 완료:", submission.shape)