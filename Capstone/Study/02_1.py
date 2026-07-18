import numpy as np
import matplotlib.pyplot as plt
x_data = [1,2,3,4,5,6,7,8]
y_data = [40,50,55,58,65,70,75,80]

w = 1.0

def forward(x):
    return x * w

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

w_list = []
mse_list = []

for w in np.arange(0.0,100.0,1.0):
    print("w=",w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        l = loss(x_val, y_val)
        l_sum += l
        print("\t", x_val, y_val, y_pred_val, l)
    print("MSE=",l_sum/8)
    w_list.append(w)
    mse_list.append(l_sum/8)

plt.plot(w_list,mse_list)
plt.ylabel("Loss")
plt.xlabel("w")
plt.show()