# ===== 라이브러리 =====
import torch.nn as nn


class HousePriceModel(nn.Module):
    def __init__(self, input_dim, p=0.0):                  # ★[v5] p 추가: dropout 비율 (0.0이면 규제 없음=기존과 동일)
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)               # 입력 -> 은닉1
        self.fc2 = nn.Linear(128, 64)                      # 은닉1 -> 은닉2
        self.fc3 = nn.Linear(64, 1)                        # 은닉2 -> 출력
        self.relu = nn.ReLU()                              # 활성화 함수
        self.drop = nn.Dropout(p)                          # ★[v5] dropout 층 추가: 학습 중 뉴런 일부를 랜덤으로 끔(과적합 억제)

    def forward(self, x):
        x = self.drop(self.relu(self.fc1(x)))              # ★[v5] 1층 ReLU 뒤에 dropout 적용
        x = self.drop(self.relu(self.fc2(x)))              # ★[v5] 2층 ReLU 뒤에 dropout 적용
        x = self.fc3(x)                                    # 출력층(활성화 없음)
        return x