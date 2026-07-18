x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w1 = 1.0
w2 = 1.0
b = 0
learning_rate = 0.001

def forward(x):
    return x * x * w2 + x * w1 + b

def loss(x,y):
    y_pred = forward(x)
    return(y_pred - y) ** 2

def gradient(x,y):
    y_pred = forward(x)
    error = y_pred - y

    grad_w1 = 2 * error * x
    grad_w2 = 2 * error * x ** 2
    grad_b = 2 * error

    return grad_w1, grad_w2, grad_b

print("predict (before training)", 4, forward(4))

for epoch in range(1000):

    for x_val, y_val in zip(x_data, y_data):

        grad_w1, grad_w2, grad_b = gradient(x_val, y_val)

        # 각각의 기울기를 이용해 별도로 업데이트
        w1 = w1 - learning_rate * grad_w1
        w2 = w2 - learning_rate * grad_w2
        b = b - learning_rate * grad_b

    if epoch % 100 == 0:
        print(
            "epoch:", epoch,
            "w1:", w1,
            "w2:", w2,
            "b:", b,
            "loss:", loss(x_val, y_val)
        )

print("Predict after training:", forward(4))
"""

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = 1.0

def forward(x):
    return x * w

def loss(x,y):
    y_pred = forward(x)
    return(y_pred - y) ** 2

def gradient(x,y):
    return 2 * x * (x * w - y)

print("predict (before training)", 4, forward(4))

for epoch in range(100):
    for x_val, y_val in zip(x_data,y_data):
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad
        print("\tgrad: ", x_val, y_val, grad)
        l = loss(x_val,y_val)

    print("progress: ",epoch,"w=",w,"loss=",l)

print("Predict (after training)", "4hours",forward(4))
"""