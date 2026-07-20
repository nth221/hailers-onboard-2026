import torch
import torch.nn as nn
import pandas as pd

from preprocess import HousePriceData
from model import HousePriceModel

torch.manual_seed(42)                                       # 재현성 고정

# ===== 데이터 준비 (★ 타깃 인코딩) =====
X_train, X_val, y_train, y_val, X_test, test_ids = \
    HousePriceData(encoding="target").prepare()

# ===== 모델 / loss / optimizer (규제 없음, v3와 동일) =====
model = HousePriceModel(input_dim=X_train.shape[1])         # p 기본값 0.0 → dropout 없음
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ===== training / validation loop (+ early stopping / best checkpoint) =====
epochs = 3000                                               # 변수 ~80개라 넉넉히 (early stopping이 멈춤)
best_val = float("inf")
patience = 50
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

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:4d} | train: {loss.item():.4f} | val: {val_loss:.4f} | val RMSE: {val_loss**0.5:.4f}")

    if counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1} | best val: {best_val:.4f} (RMSE {best_val**0.5:.4f})")
        break

# ===== 최고 모델 불러오기 =====
model.load_state_dict(torch.load("best_model.pth"))

# ===== test 예측 → 제출 파일 생성 (이기면 주석 풀어서 제출) =====
#model.eval()
#with torch.no_grad():
#    pred_log = model(X_test)
#pred = torch.expm1(pred_log).squeeze().numpy()
#submission = pd.DataFrame({"Id": test_ids, "SalePrice": pred})
#submission.to_csv("submission.csv", index=False)
#print("submission.csv 저장 완료:", submission.shape)
#print(submission.head())