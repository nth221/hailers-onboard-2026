# PyTorch를 불러옵니다.
import torch

# CSV 숫자 데이터를 읽기 위해 NumPy를 불러옵니다.
import numpy as np

# 나만의 데이터셋을 만들기 위한 Dataset,
# 데이터를 여러 묶음으로 전달하기 위한 DataLoader를 불러옵니다.
from torch.utils.data import Dataset, DataLoader
import gzip

# =========================================================
# 1. 사용자 정의 Dataset
# =========================================================

# Dataset을 상속받아 당뇨병 데이터셋 클래스를 만듭니다.
class DiabetesDataset(Dataset):

    # 데이터셋 객체를 만들 때 한 번 실행됩니다.
    def __init__(self):

        # CSV 파일을 읽습니다.
        # 각 값은 쉼표로 구분되어 있고 float32 형태로 읽습니다.
        with gzip.open("diabetes.csv", "rt") as file:
            xy = np.loadtxt(
                file,
                delimiter=",",
                dtype=np.float32
            )

        # 전체 데이터 개수를 저장합니다.
        # shape[0]은 행의 개수, 즉 환자 수입니다.
        self.len = xy.shape[0]

        # 모든 행에서 마지막 열을 제외한 열을 가져옵니다.
        # 환자 한 명당 입력 변수가 8개라고 가정합니다.
        self.x_data = torch.from_numpy(
            xy[:, 0:-1]
        )

        # 모든 행에서 마지막 열 하나만 가져옵니다.
        # 마지막 열은 당뇨 여부인 0 또는 1입니다.
        #
        # [-1]처럼 리스트로 작성하면
        # 결과 모양이 [데이터 수, 1]로 유지됩니다.
        self.y_data = torch.from_numpy(
            xy[:, [-1]]
        )

    # index번째 데이터 하나를 꺼낼 때 실행됩니다.
    def __getitem__(self, index):

        # 입력 데이터와 정답을 한 쌍으로 반환합니다.
        return self.x_data[index], self.y_data[index]

    # 데이터셋의 전체 데이터 개수를 반환합니다.
    def __len__(self):

        return self.len


# 당뇨병 데이터셋 객체를 만듭니다.
dataset = DiabetesDataset()


# =========================================================
# 2. DataLoader
# =========================================================

# 데이터셋의 데이터를 배치 단위로 공급합니다.
train_loader = DataLoader(

    # 사용할 데이터셋
    dataset=dataset,

    # 한 번에 32명씩 학습
    batch_size=32,

    # 매 epoch마다 데이터 순서를 무작위로 섞음
    shuffle=True,

    # Mac에서 공부할 때는 우선 0으로 설정하는 것이 안전합니다.
    num_workers=0
)


# =========================================================
# 3. 신경망 모델
# =========================================================

# nn.Module을 상속받아 모델을 정의합니다.
class Model(torch.nn.Module):

    # 모델이 생성될 때 한 번 실행됩니다.
    def __init__(self):

        # 부모 클래스인 nn.Module을 초기화합니다.
        super().__init__()

        # 첫 번째 계층:
        # 입력 변수 8개를 받아 6개의 값을 만듭니다.
        self.l1 = torch.nn.Linear(8, 6)

        # 두 번째 계층:
        # 6개의 값을 받아 4개의 값을 만듭니다.
        self.l2 = torch.nn.Linear(6, 4)

        # 세 번째 계층:
        # 4개의 값을 받아 최종 결과 1개를 만듭니다.
        self.l3 = torch.nn.Linear(4, 1)

        # 모든 값을 0과 1 사이로 변환합니다.
        self.sigmoid = torch.nn.Sigmoid()

    # 데이터가 모델을 통과하는 순서입니다.
    def forward(self, x):

        # 입력 8개 → Linear → Sigmoid → 값 6개
        out1 = self.sigmoid(
            self.l1(x)
        )

        # 값 6개 → Linear → Sigmoid → 값 4개
        out2 = self.sigmoid(
            self.l2(out1)
        )

        # 값 4개 → Linear → Sigmoid → 당뇨 확률 1개
        y_pred = self.sigmoid(
            self.l3(out2)
        )

        # 최종 예측 확률을 반환합니다.
        return y_pred


# 모델 객체를 생성합니다.
model = Model()


# =========================================================
# 4. 손실함수와 Optimizer
# =========================================================

# 0과 1을 분류하는 이진 분류 손실함수입니다.
# reduction="mean"은 배치 안의 손실 평균을 계산합니다.
criterion = torch.nn.BCELoss(
    reduction="mean"
)

# SGD를 사용해 모델의 가중치와 편향을 수정합니다.
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)


# =========================================================
# 5. 학습
# =========================================================

# 전체 데이터셋을 2번 반복해서 학습합니다.
for epoch in range(2):

    # train_loader가 데이터를 32개씩 전달합니다.
    # i는 현재 몇 번째 배치인지 나타냅니다.
    for i, data in enumerate(train_loader):

        # data 안에는 입력과 정답이 들어 있습니다.
        inputs, labels = data

        # Forward:
        # 현재 가중치로 당뇨 확률을 예측합니다.
        y_pred = model(inputs)

        # 예측 확률과 실제 정답을 비교해 손실을 계산합니다.
        loss = criterion(y_pred, labels)

        # 이전 배치에서 계산된 기울기를 초기화합니다.
        optimizer.zero_grad()

        # Backward:
        # 각 가중치의 기울기를 계산합니다.
        loss.backward()

        # 계산한 기울기를 이용해 가중치를 수정합니다.
        optimizer.step()

        # 현재 epoch, 배치 번호, 손실을 출력합니다.
        print(
            "epoch:", epoch,
            "batch:", i,
            "loss:", loss.item()
        )





