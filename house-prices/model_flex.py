import torch.nn as nn


class HousePriceModelFlex(nn.Module):
    """은닉층 크기를 자유롭게 지정 가능한 MLP.
       hidden=(64,16)이면 기존 model2와 동일한 구조."""
    def __init__(self, input_dim, hidden=(64, 16), p=0.1):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)