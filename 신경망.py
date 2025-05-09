# -*- coding: utf-8 -*-
"""
Created on Thu May  1 17:50:32 2025

@author: pc
"""
import numpy as np
import matplotlib.pylab as plt

# 계단 함수 구현

def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
    
i = step_function(3.0)
print(i)
# 넘파이 배열은 지원안됌
#j = step_function(np.array([1.0, 2.0]))

# 넘파이 지원 계단 함수
def step_function1(x):
    y = x > 0
    return y.astype(np.int)

x = np.array([-1.0, 1.0, 2.0])
print(x)

y = x>0
print(y)

def step_function2(x):
    return np.array(x > 0, dtype = int)

x = np.arange(-0.5, 0.5, 0.1)
y = step_function2(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# 시그모이드 함수

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))


t = np.array([1.0, 2.0, 3.0])
print(1.0 + t)

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# ReLU
def relu(x):
    return np.maximun(0, x)

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + 1

Z1 = sigmoid(A1)

print(A1)
print(Z1)

W2 = np.array([[0.1,0.4], [0.2,0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

def identity_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)
print(Y)

def init_network():
    network = {}
    network['W1']