# House Prices - Advanced Regression Techniques

## PyTorch MLP 기반 주택 가격 예측 프로젝트

> Kaggle의 주택 특성 데이터를 이용하여 주택 판매 가격(`SalePrice`)을 예측하는 회귀 프로젝트이다.  
> 모델은 `torch.nn.Module`로 직접 정의했으며, 전처리·학습·검증·Early Stopping·5-Fold 앙상블을 직접 구현했다.

---

## 1. 프로젝트 목표

### 문제 정의

- 입력: 주택의 면적, 품질, 건축 연도, 지역, 차고, 지하실 등 다양한 특성
- 출력: 주택의 실제 판매 가격인 `SalePrice`
- 문제 유형: 지도학습 기반 회귀 문제
- 평가 지표: 로그 가격 기준 RMSE

### 성능 목표

| 구분 | 목표 RMSE |
|---|---:|
| Required | 0.1233 미만 |
| Ambitious | 0.1199 미만 |
| 현재 최고 Kaggle 점수 | **0.12314** |

현재 모델은 Required 기준을 통과했으며, Ambitious 목표까지 약 `0.00324`의 추가 개선이 필요하다.

### 과제 정책

- `torch.nn.Module`로 모델 직접 정의
- 최소 2개 이상의 hidden layer 사용
- optimizer와 loss 직접 설정
- training loop와 validation loop 직접 작성
- Early Stopping 또는 best checkpoint 구현
- Tree Ensemble, AutoML, 공개 Notebook 전체 복사는 최종 모델로 사용하지 않음

---

## 2. 데이터 구성

### 데이터 크기

| 데이터 | 행 | 열 | 역할 |
|---|---:|---:|---|
| `train.csv` | 1,460 | 81 | 모델 학습 및 검증 |
| `test.csv` | 1,459 | 80 | Kaggle 제출용 예측 |

`train.csv`에는 정답인 `SalePrice`가 포함되어 있고, `test.csv`에는 포함되어 있지 않다.

### 입력 데이터 유형

- 수치형: 면적, 방 개수, 연도, 품질 점수 등
- 범주형: 지역, 주택 형태, 지붕 종류, 차고 형태 등
- 식별자: `Id`
- 정답: `SalePrice`

### 주요 특성 예시

| 영역 | 주요 변수 | 의미 |
|---|---|---|
| 주택 품질 | `OverallQual`, `OverallCond` | 전체 품질과 상태 |
| 면적 | `GrLivArea`, `TotalBsmtSF`, `1stFlrSF` | 생활 및 층별 면적 |
| 연도 | `YearBuilt`, `YearRemodAdd`, `YrSold` | 건축·리모델링·판매 연도 |
| 욕실 | `FullBath`, `HalfBath`, `BsmtFullBath` | 지상 및 지하실 욕실 |
| 차고 | `GarageCars`, `GarageArea`, `GarageType` | 차고 크기와 종류 |
| 지역 | `Neighborhood` | 주택이 위치한 지역 |

---

## 3. 데이터 전처리

신경망에 입력하려면 모든 값이 숫자여야 하며 결측값이 없어야 한다.

### 3.1 입력과 정답 분리

```python
y = train["SalePrice"]
x = train.drop(columns=["Id", "SalePrice"])

test_id = test["Id"]
x_test = test.drop(columns=["Id"])
```

`Id`는 행을 구분하기 위한 번호이므로 학습 입력에서는 제거하고, 제출 파일을 만들 때 다시 사용한다.

### 3.2 결측값 처리

| 데이터 유형 | 처리 방법 | 이유 |
|---|---|---|
| 수치형 | Fold 학습 데이터의 중앙값 | 극단값의 영향을 평균보다 적게 받음 |
| 범주형 | `Missing` 문자열 | 값이 없다는 상태 자체를 하나의 범주로 보존 |

중요한 점은 전체 데이터의 중앙값을 미리 사용하지 않는 것이다. 각 Fold의 학습 데이터에서만 중앙값을 계산하여 검증 데이터의 정보가 학습에 들어가는 데이터 누수를 방지했다.

### 3.3 수치형 스케일링

```python
StandardScaler()
```

면적은 수천 단위, 욕실 개수는 한 자릿수처럼 변수마다 크기가 다르다. StandardScaler는 수치형 변수들의 평균을 0, 표준편차를 1에 가깝게 만들어 신경망 학습을 안정화한다.

### 3.4 범주형 인코딩

기본 MLP에서는 다음 인코딩을 사용했다.

```python
OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)
```

문자 범주를 0과 1로 구성된 숫자 열로 변환하며, 검증 또는 테스트에서 처음 등장한 범주는 무시하여 오류를 방지한다.

### 3.5 올바른 `fit`과 `transform`

```python
x_train_processed = preprocessor.fit_transform(x_train_fold)
x_valid_processed = preprocessor.transform(x_valid_fold)
x_test_processed = preprocessor.transform(x_test_fold)
```

- `fit_transform`: 학습 데이터에서 전처리 기준을 배우고 적용
- `transform`: 학습 데이터에서 배운 기준을 검증 및 테스트 데이터에 그대로 적용
- 검증 및 테스트 데이터에 `fit_transform`을 사용하면 데이터 누수가 발생할 수 있음

---

## 4. Feature Engineering

원본 변수에서 주택 가격과 관련된 정보를 더 직접적으로 표현하는 두 개의 파생 변수를 추가했다.

### 4.1 HouseAge

```python
data["HouseAge"] = data["YrSold"] - data["YearBuilt"]
```

`YearBuilt`만으로는 판매 당시 집이 몇 년 된 주택인지 모델이 바로 알기 어렵다. 판매 연도에서 건축 연도를 빼서 판매 당시의 실제 주택 나이를 표현했다.

### 4.2 TotalBath

```python
data["TotalBath"] = (
    data["FullBath"].fillna(0)
    + 0.5 * data["HalfBath"].fillna(0)
    + data["BsmtFullBath"].fillna(0)
    + 0.5 * data["BsmtHalfBath"].fillna(0)
)
```

욕실 정보가 지상·지하, 전체·반쪽 욕실로 흩어져 있기 때문에 하나의 총 욕실 수로 요약했다. Half Bath는 0.5개로 계산했다.

### Feature Engineering 해석

파생 변수는 원본에 없던 정보를 새로 만드는 것만을 의미하지 않는다. 여러 열에 흩어진 관계를 모델이 이해하기 쉬운 한 열로 정리하는 것도 Feature Engineering이다.

---

## 5. 타깃 로그 변환

```python
y_log = np.log1p(y)
```

주택 가격은 일부 고가 주택 때문에 오른쪽 꼬리가 긴 분포를 가진다. 원래 가격을 그대로 학습하면 비싼 주택의 절대 오차가 학습에 지나치게 큰 영향을 줄 수 있다.

로그 변환의 효과는 다음과 같다.

- 고가 주택과 일반 주택의 가격 차이를 압축
- 극단적으로 비싼 집의 영향 완화
- Kaggle 평가 방식과 동일한 로그 단위에서 학습
- 가격의 절대 차이보다 비율 차이에 가까운 오차를 학습

예측이 끝난 후에는 다음 코드로 실제 가격 단위로 복구한다.

```python
prediction_price = np.expm1(prediction_log)
```

---

## 6. 타깃 중심화

각 Fold에서 학습 정답의 평균을 빼서 정답이 0 근처에 위치하도록 만들었다.

```python
target_mean = float(y_train_fold.mean())

y_train_centered = y_train_fold - target_mean
y_valid_centered = y_valid_fold - target_mean
```

### 중심화 전후

```text
기존 로그 가격: 약 11.0 ~ 13.5
중심화 가격:    약 -1.0 ~ +1.5
```

신경망은 처음부터 약 12라는 큰 출력값을 만드는 대신, 평균 가격보다 얼마나 비싸거나 저렴한지만 학습하면 된다.

예측 후에는 학습 시 제거한 평균을 다시 더한다.

```python
valid_log_predictions = valid_centered_predictions + target_mean
test_log_predictions = test_centered_predictions + target_mean
```

중요하게 각 Fold의 `target_mean`은 해당 Fold의 학습 정답만 사용하여 계산했다.

---

## 7. 기본 MLP 모델

### 모델 구조

```text
입력 특성
   ↓
Linear(input_dim → 128)
   ↓
ReLU + Dropout(0.1)
   ↓
Linear(128 → 64)
   ↓
ReLU + Dropout(0.1)
   ↓
Linear(64 → 1)
   ↓
중심화된 로그 집값
```

### 모델 코드

```python
class HousePriceMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x.squeeze(1)
```

### 주요 구성 요소

- `Linear`: 입력값을 가중합하여 새로운 표현 생성
- `ReLU`: 모델이 비선형 관계를 학습할 수 있도록 함
- `Dropout`: 학습 중 일부 뉴런을 무작위로 끄고 특정 뉴런 의존을 줄임
- 마지막 출력 1개: 주택 한 채의 로그 가격 예측

---

## 8. 학습 설정

### Loss

```python
criterion = nn.MSELoss()
```

예측과 정답의 차이를 제곱한 평균이다. 큰 오차에 더 큰 벌점을 준다.

### Optimizer

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)
```

- `model.parameters()`: 수정할 모델 가중치
- `lr`: 한 번의 업데이트에서 움직이는 정도
- `weight_decay`: 가중치가 지나치게 커지는 것을 억제하여 과적합 완화

### Mini-batch

```python
batch_size = 32
```

전체 학습 데이터를 한꺼번에 처리하지 않고 32개 주택씩 나누어 가중치를 업데이트했다. 메모리 사용을 줄이고 업데이트 횟수를 늘려 학습을 진행한다.

### 학습 Loop 핵심

```python
optimizer.zero_grad()
predictions = model(batch_x)
loss = criterion(predictions, batch_y)
loss.backward()
optimizer.step()
```

1. 이전 gradient 초기화
2. 모델 예측
3. 오차 계산
4. 역전파로 각 가중치의 gradient 계산
5. optimizer가 gradient를 이용해 가중치 수정

---

## 9. Validation과 Early Stopping

검증 중에는 가중치를 수정하지 않는다.

```python
model.eval()

with torch.no_grad():
    predictions = model(batch_x)
```

- `model.eval()`: Dropout 등을 평가 모드로 변경
- `torch.no_grad()`: gradient를 계산하지 않아 메모리와 계산량 절약

검증 RMSE가 개선될 때만 모델을 저장한다.

```python
if valid_rmse < best_valid_rmse:
    best_valid_rmse = valid_rmse
    patience_counter = 0
    torch.save(model.state_dict(), checkpoint_path)
else:
    patience_counter += 1
```

20 Epoch 동안 개선되지 않으면 학습을 종료했다.

```python
if patience_counter >= 20:
    break
```

마지막 Epoch의 모델이 아니라 검증 성능이 가장 좋았던 checkpoint를 다시 불러와 예측했다.

---

## 10. 5-Fold Cross Validation

### 적용 이유

학습 데이터가 1,460개로 적기 때문에 한 번의 train/validation 분할 결과는 어떤 집이 검증 데이터에 포함되었는지에 따라 크게 달라질 수 있다.

5-Fold는 전체 데이터를 다섯 부분으로 나누고, 매번 한 부분을 검증 데이터로 사용한다.

```text
Fold 1: [Valid][Train][Train][Train][Train]
Fold 2: [Train][Valid][Train][Train][Train]
Fold 3: [Train][Train][Valid][Train][Train]
Fold 4: [Train][Train][Train][Valid][Train]
Fold 5: [Train][Train][Train][Train][Valid]
```

모든 데이터는 정확히 한 번 검증에 사용되며 네 번 학습에 사용된다.

### OOF Prediction

각 주택에 대해 해당 주택을 학습하지 않은 모델의 예측값을 저장했다. 이것을 Out-of-Fold Prediction이라고 한다.

```python
oof_predictions[valid_idx] = valid_log_predictions
```

전체 OOF RMSE는 모든 학습 행에 대한 일반화 성능을 하나의 값으로 나타낸다.

### 테스트 예측 앙상블

다섯 개 모델이 각각 테스트 데이터의 가격을 예측하고 로그 예측값을 평균했다.

```python
test_log_predictions = np.mean(
    np.stack(test_fold_predictions),
    axis=0
)
```

하나의 데이터 분할에서 발생하는 우연한 오차를 다섯 모델의 평균으로 완화한다.

---

## 11. 평가 지표: RMSE

RMSE는 Root Mean Squared Error의 약자이다.

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{y_i}-y_i)^2}
$$

- 예측값과 정답의 차이를 제곱
- 전체 데이터의 평균 계산
- 마지막에 제곱근 적용
- 값이 낮을수록 좋은 모델

이 프로젝트에서는 `log1p(SalePrice)` 단위의 RMSE를 사용한다.

---

## 12. 최종 5-Fold 검증 결과

| Fold | Best RMSE |
|---:|---:|
| 1 | 0.13329 |
| 2 | 0.11911 |
| 3 | 0.18482 |
| 4 | 0.12481 |
| 5 | 0.10480 |
| Fold 평균 | 0.13337 |
| 전체 OOF | **0.13614** |

### 결과 해석

- Fold 2와 Fold 5에서는 목표 수준의 성능이 확인됨
- Fold 3의 오차가 `0.18482`로 매우 높음
- Fold별 차이가 크므로 특정 종류의 주택에서 모델이 불안정한 것으로 판단
- 단순히 모델 크기를 늘리기보다 Fold 3과 OOF 오차가 큰 주택을 분석할 필요가 있음

### Kaggle 결과

| 모델 단계 | Kaggle RMSE |
|---|---:|
| Dropout 및 TotalBath 수정 단계 | 0.15135 |
| Feature Engineering 개선 단계 | 0.14745 |
| 타깃 중심화 + 5-Fold 앙상블 | **0.12314** |

Kaggle 점수 `0.12314`로 Required 목표인 `0.1233 미만`을 달성했다.

OOF와 Kaggle 점수가 정확히 같지 않은 이유는 Kaggle 테스트 데이터와 학습 데이터의 구성 및 난이도가 다르기 때문이다.

---

## 13. 실패하거나 폐기한 실험

### 13.1 무조건적인 파생 변수 추가

여러 면적을 합친 `TotalSF` 등의 파생 변수를 추가했지만 항상 성능이 좋아지지는 않았다.

가능한 원인은 다음과 같다.

- 원본 면적 변수와 정보가 중복됨
- MLP가 이미 원본 변수의 조합을 학습하고 있었음
- 데이터가 적어 불필요한 변수가 과적합을 증가시킴
- 파생 변수의 정의가 실제 가격 구조를 충분히 반영하지 못함

따라서 파생 변수는 많을수록 좋은 것이 아니며, 하나씩 추가하고 OOF 성능으로 판단해야 한다.

### 13.2 이상치 행 삭제

면적이 크지만 가격이 낮은 주택과 같은 이상치를 제거했으나 Kaggle 성능이 하락했다.

가능한 원인은 다음과 같다.

- 이상치처럼 보이는 주택이 테스트 데이터에도 존재할 수 있음
- 1,460개뿐인 데이터에서 행 삭제로 학습 정보가 더 감소함
- 특이한 주택도 실제로 존재하는 유효한 데이터였을 수 있음

결론적으로 이상치를 무조건 삭제하는 방식은 폐기했다.

### 13.3 Hidden Layer 크기 변경

Hidden Layer를 크게 만든다고 성능이 자동으로 향상되지는 않았다. 데이터가 적은 상황에서는 모델이 커질수록 학습 데이터를 암기해 과적합될 수 있다.

---

## 14. 현재 모델의 한계

### 14.1 작은 데이터

학습 데이터가 1,460개뿐인데 One-Hot Encoding 후 입력 차원은 약 300개이다. 입력 차원에 비해 행 수가 적어 신경망이 과적합되기 쉽다.

### 14.2 Fold별 성능 편차

Fold 3과 Fold 5의 RMSE 차이가 약 `0.08`이다. 특정 분할 또는 특정 주택 유형에 모델이 민감하다는 의미이다.

### 14.3 One-Hot Encoding의 한계

One-Hot Encoding은 범주 사이의 관계를 직접 표현하지 못한다. 예를 들어 서로 비슷한 지역도 완전히 독립적인 열로 표현된다.

### 14.4 제한적인 피처 상호작용

MLP가 피처 관계를 학습할 수는 있지만, 어떤 피처가 어떤 피처를 참고해야 하는지 명시적으로 모델링하지 않는다.

---

## 15. 다음 실험 계획

아래 내용은 현재까지 완료한 최종 결과가 아니라 앞으로 검증할 후보이다.

### 15.1 Huber Loss

현재 MSE는 큰 오차를 제곱하기 때문에 일부 어려운 주택의 영향을 크게 받는다.

```python
criterion = nn.HuberLoss(delta=0.1)
```

Huber Loss는 작은 오차에서는 MSE처럼 작동하고, 큰 오차에서는 영향이 선형적으로 증가한다.

#### 실험 가설

> 이상치 행을 삭제하지 않으면서, 일부 큰 오차가 학습을 지나치게 흔드는 현상만 완화하면 전체 성능과 Fold 3 성능이 개선될 수 있다.

#### 주의점

Huber Loss의 제곱근은 RMSE가 아니다. 학습은 Huber Loss를 사용하되, 검증 및 Early Stopping은 별도로 계산한 RMSE를 사용해야 한다.

### 15.2 FT-Transformer

FT-Transformer는 각 열을 하나의 토큰으로 변환하고 Attention을 이용해 피처 간 관계를 학습하는 표 데이터 전용 Transformer이다.

```text
수치형 → Numeric Token ─────┐
                            ├→ Transformer → Regression Head → 가격
범주형 → Embedding Token ───┘
```

기본 MLP와 달리 범주형 피처에 One-Hot Encoding 대신 Embedding을 사용한다.

#### 초기 실험 구조

- Token dimension: 64
- Attention heads: 8
- Transformer layers: 2
- Dropout: 0.1
- Regression head: `64 → 32 → 1`
- Loss: MSE
- Validation: 동일한 5-Fold

첫 실험에서는 Huber Loss를 동시에 적용하지 않는다. FT-Transformer 구조 자체의 효과를 MLP와 공정하게 비교한 후, 효과가 있을 때 Huber Loss를 결합한다.

### 15.3 Optuna

Optuna는 모델이 아니라 하이퍼파라미터 탐색 도구이다.

탐색 후보는 다음과 같다.

- learning rate
- weight decay
- dropout
- Huber delta

모델 구조와 학습·검증 루프는 직접 구현하고, 숫자 조합 탐색만 Optuna에 맡기는 방식이다. 과제의 AutoML 제한에 포함되는지는 교수자 확인이 필요하다.

### 15.4 TabPFN

TabPFN은 수많은 합성 표 데이터로 미리 학습된 Foundation Model이다. House Prices와 같은 작은 표 데이터의 참고 성능을 빠르게 확인하는 용도로 사용할 수 있다.

하지만 사전학습 모델을 불러와 사용하므로 `nn.Module`, optimizer, training loop를 직접 구현해야 하는 이번 과제의 최종 모델로는 적절하지 않다.

따라서 TabPFN은 최종 모델이 아니라 참고 베이스라인으로만 고려한다.

### 15.5 ExcelFormer

ExcelFormer는 표 데이터에 특화된 발전된 Attention 구조를 사용한다. 그러나 구현 난이도가 높고 작은 데이터에서 과적합 가능성이 있으며 공개 코드를 과도하게 의존할 위험도 있다.

현재 프로젝트에서는 구현 우선순위보다 논문 및 발전 방향 조사 대상으로 둔다.

---

## 16. 앞으로의 실험 순서

```text
현재 최고 모델 보존
MLP + MSE + 5-Fold + 타깃 중심화
Kaggle RMSE = 0.12314
          ↓
실험 A: MLP + Huber Loss
          ↓
전체 OOF와 Fold 3 개선 여부 확인
          ↓
실험 B: FT-Transformer + MSE
          ↓
MLP와 동일한 5-Fold에서 비교
          ↓
효과가 확인되면 FT-Transformer + Huber
          ↓
필요시 제한된 Optuna 탐색
          ↓
OOF가 확실히 개선된 모델만 Kaggle 제출
```

---

## 17. 실험 관리 방법

Kaggle 제출 횟수가 제한되어 있으므로 Public Score를 실험 도구처럼 반복해서 사용하지 않는다. 모델 선택은 먼저 OOF RMSE를 기준으로 수행한다.

| 실험 | 가설 | 한 가지 변경 | OOF | Fold 3 | Kaggle | 결정 |
|---|---|---|---:|---:|---:|---|
| Baseline | 기준 모델 | MLP + MSE | 0.13614 | 0.18482 | 0.12314 | 유지 |
| Huber | 큰 오차 영향 완화 | Loss만 변경 | 측정 예정 | 측정 예정 | 미제출 | OOF 비교 |
| FT-Transformer | 피처 관계 학습 | 모델·인코딩 변경 | 측정 예정 | 측정 예정 | 미제출 | OOF 비교 |

### 실험 원칙

1. 한 번에 하나의 가설만 검증한다.
2. 모든 실험에서 같은 5-Fold와 seed를 사용한다.
3. 전처리기는 각 Fold의 train 데이터에만 fit한다.
4. 전체 OOF뿐 아니라 Fold별 점수와 표준편차도 비교한다.
5. OOF가 나빠진 모델은 Kaggle에 제출하지 않는다.
6. OOF가 확실히 개선된 모델만 제한적으로 제출한다.
7. 성공한 실험뿐 아니라 실패한 실험과 원인도 기록한다.

---

## 18. 결론

이번 프로젝트에서는 단순한 MLP에서 시작하여 데이터 전처리, Feature Engineering, 로그 변환, Dropout, 타깃 중심화, Early Stopping, 5-Fold Cross Validation 및 앙상블을 단계적으로 적용했다.

최종적으로 Kaggle RMSE `0.12314`를 기록하여 Required 목표인 `0.1233 미만`을 달성했다.

성능 향상에 가장 큰 영향을 준 부분은 단순히 모델을 크게 만드는 것이 아니라 다음 과정이었다.

- 올바른 결측값 처리와 데이터 누수 방지
- `HouseAge`, `TotalBath`와 같은 의미 있는 파생 변수
- 로그 타깃 변환과 타깃 중심화
- 단일 분할 대신 5-Fold 검증
- 다섯 모델의 테스트 예측 앙상블
- Early Stopping을 통한 과적합 방지

현재 모델은 Fold별 편차가 크다는 한계가 있다. 다음 단계에서는 이상치를 삭제하는 대신 Huber Loss로 큰 오차의 영향을 완화하고, FT-Transformer를 이용해 피처 간 상호작용을 더 직접적으로 학습하는 실험을 진행할 예정이다.

---

## 19. 예상 질문과 답변

### Q1. 왜 SalePrice에 로그를 적용했나요?

고가 주택 때문에 가격 분포가 한쪽으로 치우쳐 있기 때문이다. 로그 변환으로 큰 가격 차이를 압축하여 학습을 안정화하고 Kaggle의 로그 RMSE 평가 방식과 맞췄다.

### Q2. 왜 결측값을 전체 데이터에서 한 번에 처리하지 않았나요?

전체 데이터의 중앙값이나 범주 정보를 사용하면 검증 데이터의 정보가 학습 과정에 포함되는 데이터 누수가 발생할 수 있다. 따라서 각 Fold의 학습 데이터에서만 기준을 계산했다.

### Q3. 왜 5-Fold를 사용했나요?

데이터가 1,460개로 적기 때문에 한 번의 분할 결과가 불안정할 수 있다. 모든 데이터를 검증에 한 번씩 사용하여 일반화 성능을 더 안정적으로 측정하고, 다섯 모델의 평균으로 테스트 예측의 분산도 줄였다.

### Q4. 왜 이상치를 제거하지 않았나요?

실제 제거 실험에서 Kaggle 점수가 나빠졌다. 이상치처럼 보이는 데이터도 실제로 존재하는 주택일 수 있고, 작은 데이터에서 행 삭제는 유용한 정보 손실을 만들 수 있기 때문이다.

### Q5. OOF RMSE와 Kaggle RMSE가 왜 다른가요?

OOF는 train 데이터 내부에서 계산되고 Kaggle 점수는 공개되지 않은 test 정답으로 계산된다. 두 데이터의 구성과 난이도가 다르기 때문에 점수가 완전히 같을 필요는 없다.

### Q6. FT-Transformer를 사용하는 이유는 무엇인가요?

기본 MLP는 모든 입력을 하나의 벡터로 처리하지만 FT-Transformer는 각 피처를 토큰으로 만들고 Attention을 통해 피처 사이의 관계를 학습한다. 지역과 품질, 연도와 주택 나이처럼 서로 연결된 관계를 더 직접적으로 표현할 가능성이 있다.

### Q7. TabPFN을 최종 모델로 사용하지 않는 이유는 무엇인가요?

TabPFN은 이미 사전학습된 Foundation Model이므로 모델·optimizer·학습 루프를 직접 구현해야 하는 과제 조건과 맞지 않는다. 따라서 참고 성능을 확인하는 베이스라인으로만 고려한다.

### Q8. 현재 가장 큰 한계는 무엇인가요?

Fold별 RMSE 편차가 크며 특히 Fold 3의 성능이 낮다. 이는 특정 주택 유형에 대한 예측이 불안정하다는 의미이므로 OOF 오차 분석과 강건한 손실함수 실험이 필요하다.

---

## 20. 발표 마무리 한 문장

> 단순히 복잡한 모델을 사용하는 것보다 데이터 누수를 방지한 전처리, 안정적인 교차검증, 의미 있는 피처 설계와 실험 관리가 주택 가격 예측 성능 향상에 더 중요하다는 것을 확인했다.
