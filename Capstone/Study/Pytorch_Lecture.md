



-2강-
loss = y^-y (y^=y예상치)
MSE = Mean square error = 0일시 완벽한 정답

-3강-
학습이란 ? 손실을 최소화하는 값을 찾는것
arg min loss
경사 하강법 : 임의의 지점에서 w기울기를 계산해서 안으로갈지 밖으로갈지(양수면 안으로)
-> w = w - a(aloss/aw) -> 미분 때리면 됌!

-4강-
Chain Rule
f = f(g); g= g(x)
df = df dg
dx   dg dx

-Logistic Regression-
sigmoid = 어떤 숫자가 와도 0~1로 바꿔주는 함수

08 데이터 로더
batch