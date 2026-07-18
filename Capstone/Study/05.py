import torch

# 데이터: y = 2x
x_data = torch.tensor([
    [1.0],
    [2.0],
    [3.0]
])

y_data = torch.tensor([
    [2.0],
    [4.0],
    [6.0]
])


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # y_pred = xw + b
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = Model()

# 평균제곱오차
criterion = torch.nn.MSELoss(reduction="mean")

# 경사하강법
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

for epoch in range(500):

    # 1. Forward
    y_pred = model(x_data)

    # 2. Loss
    loss = criterion(y_pred, y_data)

    # 3. 기존 기울기 초기화
    optimizer.zero_grad()

    # 4. Backward
    loss.backward()

    # 5. 가중치 업데이트
    optimizer.step()

    if epoch % 50 == 0:
        print(
            "epoch:", epoch,
            "loss:", loss.item(),
            "w:", model.linear.weight.item(),
            "b:", model.linear.bias.item()
        )


hour = torch.tensor([[4.0]])

with torch.no_grad():
    prediction = model(hour)

print("4시간 예측:", prediction.item())


