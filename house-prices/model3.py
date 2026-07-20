import torch.nn as nn


class HousePriceModelV3(nn.Module):
    """v7의 model2(64→16)는 풀배치 학습이 불안정하던 시기의 대응책이었음.
       미니배치+BatchNorm으로 학습이 정상화된 현재는 더 큰 용량이 적절할 수 있음."""
    def __init__(self, input_dim, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)