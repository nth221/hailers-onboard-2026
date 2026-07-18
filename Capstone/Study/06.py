import torch
import torch.nn as nn

x_data = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
y_data = torch.tensor([[0.],[0.],[1.],[1.]])

import torch.nn.functional as F
class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        z=self.linear(x)
        y_pred = self.sigmoid(z)
        return y_pred
    
model = Model()

criterion = nn.BCELoss(reduction="mean")
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch %100==0:
        print(
            "epoch:",epoch,
            "loss:",loss.item()
        )
    
model.eval()
with torch.no_grad():
    hour_1 = torch.tensor([[1.0]])
    hour_7 = torch.tensor([[7.0]])

    probability_1 = model(hour_1).item()
    probability_7 = model(hour_7).item()

    prediction_1 = probability_1 > 0.5
    prediction_7 = probability_7 > 0.5


print(
    "predict 1 hour:",
    1.0,
    "probability:",
    probability_1,
    "result:",
    prediction_1
)

print(
    "predict 7 hours:",
    7.0,
    "probability:",
    probability_7,
    "result:",
    prediction_7
)