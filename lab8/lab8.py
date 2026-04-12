import numpy as np
import matplotlib.pyplot as plt

#a1
def summation(x, w):
    return np.dot(x, w[1:]) + w[0]

def step(x): return 1 if x >= 0 else 0
def bipolar_step(x): return 1 if x >= 0 else -1
def sigmoid(x): return 1/(1+np.exp(-x))
def tanh(x): return np.tanh(x)
def relu(x): return max(0,x)
def leaky_relu(x): return x if x>0 else 0.01*x

def error(y_true, y_pred):
    return y_true - y_pred

print("A1 functions defined successfully")
'''

'''
#a2
def step(x): return 1 if x>=0 else 0

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

w = np.array([10,0.2,-0.75])
lr = 0.05
errors = []

for epoch in range(1000):
    total_error = 0
    for i in range(len(X)):
        net = np.dot(X[i], w[1:]) + w[0]
        out = step(net)
        e = y[i] - out
        total_error += e**2

        w[1:] += lr * e * X[i]
        w[0] += lr * e

    errors.append(total_error)
    if total_error <= 0.002:
        break

print("A2 Output")
print("Weights:", w)
print("Epochs:", epoch+1)

plt.plot(errors)
plt.title("Error vs Epochs")
plt.show()
'''
'''

#a3
def bipolar(x): return 1 if x>=0 else -1
def sigmoid(x): return 1/(1+np.exp(-x))
def relu(x): return max(0,x)

def train(act):
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1])
    w = np.array([10,0.2,-0.75])

    for epoch in range(1000):
        total_error = 0
        for i in range(len(X)):
            net = np.dot(X[i], w[1:]) + w[0]
            out = act(net)
            e = y[i] - out
            total_error += e**2

            w[1:] += 0.05 * e * X[i]
            w[0] += 0.05 * e

        if total_error <= 0.002:
            break

    return epoch+1

print("A3 Output")
print("Bipolar:", train(bipolar))
print("Sigmoid:", train(sigmoid))
print("ReLU:", train(relu))
'''

'''
#a4
def step(x): return 1 if x>=0 else 0

def train(lr):
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1])
    w = np.array([10,0.2,-0.75])

    for epoch in range(1000):
        total_error = 0
        for i in range(len(X)):
            net = np.dot(X[i], w[1:]) + w[0]
            out = step(net)
            e = y[i] - out
            total_error += e**2

            w[1:] += lr * e * X[i]
            w[0] += lr * e

        if total_error <= 0.002:
            break

    return epoch+1

lrs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
epochs = [train(lr) for lr in lrs]

print("A4 Output:", epochs)

plt.plot(lrs, epochs)
plt.show()
'''

'''
#a5
def step(x): return 1 if x>=0 else 0

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

w = np.array([10,0.2,-0.75])

for epoch in range(1000):
    total_error = 0
    for i in range(len(X)):
        net = np.dot(X[i], w[1:]) + w[0]
        out = step(net)
        e = y[i] - out
        total_error += e**2

        w[1:] += 0.05 * e * X[i]
        w[0] += 0.05 * e

print("A5 Output - Final Error:", total_error)
'''

'''
#a6
def sigmoid(x): return 1/(1+np.exp(-x))

X = np.array([
[20,6,2,386],[16,3,6,289],[27,6,2,393],[19,1,2,110],
[24,4,2,280],[22,1,5,167],[15,4,2,271],[18,4,2,274],
[21,1,4,148],[16,2,4,198]])

y = np.array([1,1,1,0,1,0,1,1,0,0])

X = (X - X.mean(axis=0)) / X.std(axis=0)

w = np.random.rand(5)

for epoch in range(500):
    for i in range(len(X)):
        net = np.dot(X[i], w[1:]) + w[0]
        out = sigmoid(net)
        e = y[i] - out

        w[1:] += 0.1 * e * X[i]
        w[0] += 0.1 * e

print("A6 Weights:", w)
'''

'''
#a7
X = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
y = np.array([0,0,0,1])

w = np.dot(np.linalg.pinv(X), y)

print("A7 Weights:", w)
'''

'''
#a8
def sigmoid(x): return 1/(1+np.exp(-x))

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1]).reshape(-1,1)

w1 = np.random.rand(2,2)
w2 = np.random.rand(2,1)

for epoch in range(1000):
    h = sigmoid(np.dot(X, w1))
    out = sigmoid(np.dot(h, w2))

    error = y - out

    if np.mean(error**2) <= 0.002:
        break

    d_out = error * out * (1-out)
    d_h = d_out.dot(w2.T) * h * (1-h)

    w2 += 0.05 * h.T.dot(d_out)
    w1 += 0.05 * X.T.dot(d_h)

print("A8 Epochs:", epoch+1)
'''

'''
#a9
print("A9 Output")
print("Perceptron cannot solve XOR (not linearly separable)")
'''

'''
#a10
def step(x): return 1 if x>=0 else 0

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[1,0],[1,0],[1,0],[0,1]])

w = np.random.rand(2,3)

for epoch in range(1000):
    total_error = 0
    for i in range(len(X)):
        for j in range(2):
            net = np.dot(X[i], w[j,1:]) + w[j,0]
            out = step(net)
            e = y[i][j] - out
            total_error += e**2

            w[j,1:] += 0.05 * e * X[i]
            w[j,0] += 0.05 * e

    if total_error <= 0.002:
        break

print("A10 Epochs:", epoch+1)

#a11
from sklearn.neural_network import MLPClassifier
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

model = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
model.fit(X,y)

print("A11 Accuracy:", model.score(X,y))
