import itertools
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from preprocess import HousePriceData
from model2 import HousePriceModelV2

# ===== 데이터는 한 번만 준비 =====
X_train, X_val, y_train, y_val, X_test, test_ids = HousePriceData().prepare()


def train_one(batch_size, lr, p, epochs=1000, patience=50):    # 탐색용이라 짧게
    torch.manual_seed(42)                                      # 공정 비교 위해 고정
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    model = HousePriceModelV2(input_dim=X_train.shape[1], p=p)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
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
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            break
    return best_val ** 0.5, epoch + 1


# ===== 탐색할 조합 =====
batch_sizes = [32, 64, 128]
lrs         = [0.001, 0.005, 0.01]
ps          = [0.1]                    # 우선 고정 (나중에 [0.0, 0.1, 0.2]로 확장 가능)

combos = list(itertools.product(batch_sizes, lrs, ps))
print(f"총 {len(combos)}개 조합 탐색 시작\n")

results = []
for i, (bs, lr, p) in enumerate(combos, 1):
    rmse, stopped = train_one(bs, lr, p)
    results.append((bs, lr, p, rmse, stopped))
    print(f"[{i}/{len(combos)}] batch={bs:3d} lr={lr:<6} p={p} → val RMSE {rmse:.4f} (epoch {stopped})")

# ===== 결과 정렬 (좋은 순) =====
results.sort(key=lambda r: r[3])
print("\n===== 상위 결과 =====")
print("batch |   lr   |  p  | val RMSE | epoch")
for bs, lr, p, r, e in results[:10]:
    print(f" {bs:4d} | {lr:<6} | {p} |  {r:.4f}  | {e}")

best = results[0]
print(f"\n★ 최적 조합: batch={best[0]}, lr={best[1]}, p={best[2]} → val RMSE {best[3]:.4f}")