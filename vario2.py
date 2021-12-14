import matplotlib.pyplot as plt
from math import exp, sin, cos
import numpy as np
from numpy.linalg import linalg

N = 10
D = np.zeros((N, N))

Phi = np.zeros(N)

n = 200

a, b = 0, 1
x = np.linspace(a, b, n + 1);


def phi_0(x):
    return x


def phi_0_(x):
    return 1


def phi_k(x, k):
    if k == 0:
        return phi_0(x)
    return (1 - x) * (x ** k)


def phi_k_(x, k):
    if k == 0:
        return phi_0_(x)
    return k * (x ** (k - 1)) - (k + 1) * (x ** k)


def intOne(x, i, j):
    return (x**2) * phi_k_(x, i) * phi_k_(x, j)


def intTwo(x, i, j):
    return (x**2) * phi_k(x, i) * phi_k(x, j)


def intOneTwo(x, i, j):
    return intOne(x, i, j) + intTwo(x, i, j)


for i in range(N):
    for j in range(N):
        nseg = 2 ** 10
        dx = 1.0 * (b - a) / nseg
        sum = 0.5 * (intOneTwo(a, i, j) + intOneTwo(b, i, j))

        for ii in range(1, nseg):
            sum += intOneTwo(a + ii * dx, i, j)

        D[i,j] =  sum * dx

    nseg = 2 ** 10
    dx = 1.0 * (b - a) / nseg
    sum = 0.5 * (phi_k(a,i) + phi_k(b,i))

    for ii in range(1, nseg):
        sum += phi_k(a + ii * dx,i)

    Phi[i] = sum * dx

c = linalg.solve(D, Phi)

def yyy(x):
    sum = 0
    for i in range(N):
        sum += c[i]*phi_k(x,i)
    return phi_0(x) + sum

y = np.zeros(n+1)

for i in range(n+1):
    y[i] = yyy(x[i])

plt.plot(x, y, 'r')

plt.show()