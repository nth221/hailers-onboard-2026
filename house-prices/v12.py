import copy
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from preprocess import HousePriceData
from model2 import HousePriceModelV2

# ===== 데이터 (v8 설정) =====
X_train, X_val, y_train, y_val, X_test, test_ids = HousePriceData().prepare()


def train_one(seed, batch_size=64, lr=0.005, p=0.1, epochs=2000, patience=100):
    torch.manual_seed(seed)                                    # ★ 시드만 다르게
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    model = HousePriceModelV2(input_dim=X_train.shape[1], p=p)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    counter = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())     # 최고 시점을 메모리에 보관
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            break

    # 최고 모델로 예측
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred  = model(X_val)
        test_pred = model(X_test)
    return best_val ** 0.5, val_pred, test_pred


# ===== 여러 시드로 학습 =====
seeds = [42, 0, 1, 2, 3]
val_preds, test_preds, singles = [], [], []

for s in seeds:
    rmse, vp, tp = train_one(s)
    singles.append(rmse); val_preds.append(vp); test_preds.append(tp)
    print(f"seed={s:3d} | 단일 모델 val RMSE {rmse:.4f}")

# ===== 예측 평균 = 앙상블 =====
val_avg  = torch.stack(val_preds).mean(dim=0)
test_avg = torch.stack(test_preds).mean(dim=0)
ens_rmse = torch.sqrt(((val_avg - y_val) ** 2).mean()).item()

print("\n===== 결과 =====")
print(f"단일 모델 평균 : {sum(singles)/len(singles):.4f}")
print(f"단일 모델 최고 : {min(singles):.4f}")
print(f"★ 앙상블       : {ens_rmse:.4f}")

# ===== 제출 =====
pred = torch.expm1(test_avg).squeeze().numpy()                 # 로그 공간 평균 → 가격 복원
submission = pd.DataFrame({"Id": test_ids, "SalePrice": pred})
submission.to_csv("submission.csv", index=False)
print("submission.csv 저장 완료:", submission.shape)