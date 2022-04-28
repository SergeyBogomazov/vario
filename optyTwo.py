import math

import matplotlib.pyplot as plt
from math import exp, sin, cos
import numpy as np
from numpy.linalg import linalg

def funcTwoXY(x,y):
    return 3 * (x ** 2) - 3 * x * y + 2 * (y ** 2) + x - 7 * y

def funcTwo(x):
    return 3 * (x[0] ** 2) - 3 * x[0] * x[1] + 2 * (x[1] ** 2) + x[0] - 7 * x[1]


def funcTwoX(x):
    return 6 * x[0] - 3 * x[1] + 1


def funcTwoY(x):
    return -3 * x[0] + 4 * x[1] - 7


def funcTwoZ(x):
    return funcTwoX(x) + funcTwoY(x)


x0 = np.array([1, 2])


def startPoints(n, x0, l):
    s = l * (math.sqrt(n + 1) - 1 + n) / (math.sqrt(2) * n)
    r = l * (math.sqrt(n + 1) - 1) / (math.sqrt(2) * n)

    x = np.append([x0], np.zeros((n, n)), axis=0)

    for i in range(1, n + 1):
        for j in range(0, n):
            if j + 1 == i:
                x[i][j] = x[0][j] + r
            else:
                x[i][j] = x[0][j] + s
    return x


def poly(n, f, x0, alpha, beta, gamma, lenght, epsilon):
    x = startPoints(n, x0, lenght)

    while True:
        x_applied = np.apply_along_axis(f, 1, x)
        x = x[np.argsort(x_applied)]

        c = (1 / n) * (np.sum(x, axis=0) - x[n])
        u = c + alpha * (c - x[n])

        if f(u) < f(x[0]):
            v = c + beta * (u - c)
            if f(v) < f(u):
                x[n] = v
            else:
                x[n] = u

        elif f(x[0]) <= f(u) <= f(x[n - 1]):
            x[n] = u

        elif f(x[n - 1]) <= f(u):

            if f(u) < f(x[n]):
                w = c + gamma * (u - c)
            else:
                w = c + gamma * (x[n] - c)

            if f(w) < min(f(x[n]), f(u)):
                x[n] = w
            else:
                x = (1 / 2) * (x[0] + x)

        if math.sqrt((1 / n) * sum((np.apply_along_axis(f, 1, x) - f(x[0])) ** 2)) < epsilon:
            break

    return x[0]


def gradient(f, x0, epsilonOne, lambdaV, beta, epsilon):
    x2 = x0

    while math.sqrt(abs((funcTwoX(x2) ** 2) + (funcTwoY(x2) ** 2))) >= epsilonOne:
        x1 = x2

        s1 = np.array([funcTwoX(x1), funcTwoY(x1)])
        #print("Gradient progress. x = {}, grad = {}".format(x1, s1))

        lambda_1 = lambdaV

        while funcTwo(x1 - lambda_1 * s1) - funcTwo(x1) >= -1 * epsilon * lambda_1 * np.dot(s1, s1):
            lambda_1 = beta * lambda_1

        x2 = x1 - lambda_1 * s1

    return x2


for i in range(0, 5):
    x_opt = poly(2, funcTwo, x0, alpha=1, beta=2, gamma=0.5, lenght=1, epsilon=0.01 ** i)
    print("Poly for epsilon = {} : x* = {}, f* = {}".format(0.1 ** i, x_opt, funcTwo(x_opt)))

print("="*50)

for i in range(1, 5):
    x_opt = gradient(funcTwo, x0, epsilonOne=0.1**i, lambdaV=1, beta=0.2, epsilon=0.1**i)
    print("Gradient for epsilon = {} : x* = {}, f* = {}".format(0.1 ** i, x_opt, funcTwo(x_opt)))

fig = plt.figure()
ax = plt.axes(projection='3d')

x = np.linspace(0, 2, 10)
y = np.linspace(2, 4, 10)

X, Y = np.meshgrid(x, y)
Z = funcTwoXY(X, Y)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis')

plt.show()