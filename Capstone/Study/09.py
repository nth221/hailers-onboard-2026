import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 한 번에 학습할 이미지 수
batch_size = 64

# 전체 학습 데이터 반복 횟수
epochs = 5


# ==================================================
# 1. 이미지 전처리
# ==================================================

# transforms 모듈과 이름이 겹치지 않도록
# 왼쪽 변수 이름은 transform으로 작성합니다.
transform = transforms.Compose([

    # ToTensors가 아니라 ToTensor입니다.
    # 이미지를 PyTorch Tensor로 변환합니다.
    transforms.ToTensor(),

    # MNIST 이미지 정규화
    transforms.Normalize(
        mean=(0.1307,),
        std=(0.3081,)
    )
])


# ==================================================
# 2. 학습 데이터
# ==================================================

train_dataset = datasets.MNIST(

    # 데이터를 저장할 위치
    root="./data",

    # 학습용 데이터 사용
    train=True,

    # 데이터가 없으면 자동 다운로드
    download=True,

    # 위에서 만든 전처리 적용
    transform=transform
)


# ==================================================
# 3. 테스트 데이터
# ==================================================

# 기존 코드에서 빠져 있던 부분입니다.
test_dataset = datasets.MNIST(

    root="./data",

    # False이면 테스트용 MNIST 사용
    train=False,

    download=True,

    transform=transform
)


# ==================================================
# 4. DataLoader
# ==================================================

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


# ==================================================
# 5. 신경망 모델
# ==================================================

class Net(nn.Module):

    def __init__(self):
        super().__init__()

        # MNIST 이미지의 784개 픽셀을 520개 값으로 변환
        self.l1 = nn.Linear(784, 520)

        # 520개 → 320개
        self.l2 = nn.Linear(520, 320)

        # 320개 → 240개
        self.l3 = nn.Linear(320, 240)

        # 240개 → 120개
        self.l4 = nn.Linear(240, 120)

        # 120개 → 숫자 클래스 10개
        self.l5 = nn.Linear(120, 10)

    def forward(self, x):

        # [배치, 1, 28, 28] 이미지를
        # [배치, 784] 형태로 펼칩니다.
        x = x.view(-1, 784)

        # 각 Linear 계층 뒤에 ReLU 적용
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))

        # 숫자 0~9에 대한 원래 점수 10개
        logits = self.l5(x)

        # CrossEntropyLoss를 사용하므로
        # Softmax를 직접 적용하지 않습니다.
        return logits


# 모델 생성
model = Net()


# ==================================================
# 6. 손실함수와 Optimizer
# ==================================================

# 다중 분류용 손실함수
criterion = nn.CrossEntropyLoss()

# SGD를 이용해 모델 가중치 수정
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.5
)


# ==================================================
# 7. 학습 함수
# ==================================================

def train(epoch):

    # 학습 모드
    model.train()

    # 데이터를 64개씩 가져옵니다.
    for batch_idx, (data, target) in enumerate(train_loader):

        # 이전 기울기 초기화
        optimizer.zero_grad()

        # Forward
        output = model(data)

        # 손실 계산
        loss = criterion(output, target)

        # Backward
        loss.backward()

        # 가중치 수정
        optimizer.step()

        # 100번째 배치마다 출력
        if batch_idx % 100 == 0:
            print(
                f"Epoch: {epoch} "
                f"Batch: {batch_idx}/{len(train_loader)} "
                f"Loss: {loss.item():.6f}"
            )


# ==================================================
# 8. 테스트 함수
# ==================================================

def test():

    # 평가 모드
    model.eval()

    # 맞힌 이미지 개수
    correct = 0

    # 테스트할 때는 기울기를 계산하지 않습니다.
    with torch.no_grad():

        for data, target in test_loader:

            # 클래스별 점수 계산
            output = model(data)

            # 가장 높은 점수의 클래스 선택
            prediction = output.argmax(dim=1)

            # 정답과 같은 개수 누적
            correct += (
                prediction == target
            ).sum().item()

    # 정확도 계산
    accuracy = (
        100 * correct / len(test_loader.dataset)
    )

    print(
        f"Test Accuracy: "
        f"{correct}/{len(test_loader.dataset)} "
        f"({accuracy:.2f}%)"
    )


# ==================================================
# 9. 학습 및 테스트 실행
# ==================================================

for epoch in range(1, epochs + 1):

    train(epoch)
    test()