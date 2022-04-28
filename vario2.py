import matplotlib.pyplot as plt
from math import exp, sin, cos
import numpy as np
from numpy.linalg import linalg


def Y(x):  # точное решение
    return (11 / 14) * x - (2 / (7 * (x ** 2))) + 1 / 2


N = 3
n = 100
a, b = 1, 2
h = (b - a) / n
x = np.linspace(a, b, n + 1)
y = np.zeros(n + 1)

A = np.zeros((N, N))
B = np.zeros(N)


def phi_0(x):
    return x


def phi_0_(x):
    return 1


def phi_k(x, k):
    if k == 0:
        return phi_0(x)
    return (1-x)*(2-x)*(x**(k-1))


def phi_k_(x, k):
    if k == 0:
        return phi_0(x)
    return (2*x-3)*(x**(k-1)) + (1-x)*(2-x)*(k-1)*(x**(k-1))


def intOneTwo(x, i, j):
    return (x ** 2) * phi_k_(x, i) * phi_k_(x, j) + 2 * phi_k(x, i) * phi_k(x, j)


def integrateA(i, j):
    sum = 0
    for k in range(n):
        sum += intOneTwo(x[k], i, j) * h

    A[i, j] = sum


def integrateB(i):
    sum = 0
    for k in range(n):
        sum += (phi_k(x[k], i) - intOneTwo(x[k], 0, i)) * h

    B[i] = sum


for i in range(N):
    for j in range(N):
        integrateA(i, j)
    integrateB(i)

C = linalg.solve(A, B)


def sumPhi(i):
    sum = 0
    for j in range(N):
        sum += (C[j] * phi_k(x[i], j))
    return sum


for i in range(n + 1):
    y[i] = phi_0(x[i]) + sumPhi(i)

plt.plot(x, [Y(x) for x in x], 'g')  # график точного решения
plt.plot(x, y, 'r')

plt.show()
