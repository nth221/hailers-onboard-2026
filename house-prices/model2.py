# model2.py
import torch.nn as nn


class HousePriceModelV2(nn.Module):
    def __init__(self, input_dim, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),      # 128 → 64로 축소
            nn.BatchNorm1d(64),            # ★ 배치 정규화
            nn.ReLU(),
            nn.Dropout(p),                 # ★ 약한 dropout
            nn.Linear(64, 16),             # 64 → 16으로 축소
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(16, 1),              # 출력
        )

    def forward(self, x):
        return self.net(x)