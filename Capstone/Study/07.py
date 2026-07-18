# 수치 데이터와 CSV 파일을 처리하기 위한 라이브러리
import numpy as np

# 딥러닝 계산을 위한 PyTorch
import torch

# 신경망 계층과 손실함수를 사용하기 위한 모듈
import torch.nn as nn
import gzip


# ==================================================
# 1. 데이터 불러오기
# ==================================================

# CSV 파일을 불러옵니다.
# delimiter=","는 각 값이 쉼표로 구분됐다는 뜻입니다.
# dtype=np.float32는 모든 값을 32비트 실수로 읽는다는 뜻입니다.
with gzip.open("diabetes.csv", "rt") as file:
    xy = np.loadtxt(
        file,
        delimiter=",",
        dtype=np.float32
    )


# 전체 행에서 마지막 열을 제외한 열을 입력 데이터로 사용합니다.
# 입력 변수는 총 8개입니다.
x_data = torch.from_numpy(xy[:, 0:-1])

# 전체 행에서 마지막 열만 정답 데이터로 사용합니다.
# 정답은 당뇨 여부인 0 또는 1입니다.
y_data = torch.from_numpy(xy[:, [-1]])


# 데이터 형태를 확인합니다.
print("x_data shape:", x_data.shape)
print("y_data shape:", y_data.shape)


# ==================================================
# 2. 신경망 모델 만들기
# ==================================================

# nn.Module을 상속받아 새로운 모델을 정의합니다.
class Model(nn.Module):

    # 모델을 만들 때 한 번 실행됩니다.
    def __init__(self):

        # nn.Module의 기능을 초기화합니다.
        super().__init__()

        # 첫 번째 계층
        # 입력 8개를 받아 6개의 값으로 변환합니다.
        self.l1 = nn.Linear(8, 6)

        # 두 번째 계층
        # 6개의 값을 받아 4개의 값으로 변환합니다.
        self.l2 = nn.Linear(6, 4)

        # 세 번째 계층
        # 4개의 값을 받아 최종 결과 1개를 만듭니다.
        self.l3 = nn.Linear(4, 1)

        # 결과를 0과 1 사이로 변환하는 활성화 함수입니다.
        self.sigmoid = nn.Sigmoid()

    # 입력 데이터가 모델을 통과하는 순서를 정의합니다.
    def forward(self, x):

        # 입력 8개 → 첫 번째 Linear → Sigmoid → 출력 6개
        out1 = self.sigmoid(self.l1(x))

        # 출력 6개 → 두 번째 Linear → Sigmoid → 출력 4개
        out2 = self.sigmoid(self.l2(out1))

        # 출력 4개 → 세 번째 Linear → Sigmoid → 확률 1개
        y_pred = self.sigmoid(self.l3(out2))

        # 최종 당뇨 확률을 반환합니다.
        return y_pred


# 설계한 모델의 실제 객체를 만듭니다.
model = Model()


# ==================================================
# 3. 손실함수와 Optimizer
# ==================================================

# 이진 분류용 손실함수입니다.
# 여러 데이터의 손실을 평균 내어 반환합니다.
criterion = nn.BCELoss(reduction="mean")

# SGD로 모델의 모든 가중치와 편향을 수정합니다.
# lr=0.1은 한 번에 이동하는 크기인 학습률입니다.
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)


# ==================================================
# 4. 모델 학습
# ==================================================

# 전체 데이터를 100번 반복해서 학습합니다.
for epoch in range(100):

    # Forward
    # 현재 가중치로 각 사람의 당뇨 확률을 예측합니다.
    y_pred = model(x_data)

    # 예측 확률과 실제 정답을 비교하여 손실을 계산합니다.
    loss = criterion(y_pred, y_data)

    # 이전 학습에서 남은 기울기를 지웁니다.
    optimizer.zero_grad()

    # Backward
    # 각 계층의 모든 가중치와 편향에 대한 기울기를 계산합니다.
    loss.backward()

    # 계산한 기울기로 가중치와 편향을 수정합니다.
    optimizer.step()

    # 10번마다 학습 진행 상황을 출력합니다.
    if epoch % 10 == 0:
        print(
            "epoch:", epoch,
            "loss:", loss.item()
        )


# ==================================================
# 5. 학습 결과 확인
# ==================================================

# 평가 모드로 변경합니다.
model.eval()

# 평가할 때는 기울기를 계산하지 않습니다.
with torch.no_grad():

    # 전체 데이터의 당뇨 확률을 계산합니다.
    probabilities = model(x_data)

    # 확률이 0.5 이상이면 1, 아니면 0으로 분류합니다.
    predictions = (probabilities >= 0.5).float()

    # 실제 정답과 일치하는 결과의 평균을 계산합니다.
    accuracy = (predictions == y_data).float().mean()


print("Accuracy:", accuracy.item())