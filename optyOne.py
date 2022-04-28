import matplotlib.pyplot as plt
from math import exp, sin, cos
import numpy as np
from numpy.linalg import linalg

def funcOne(x):
    return -0.8 * ((cos(x))**3) + 2.1 * exp(sin(x)) + 0.5 * x

aOne = -3.9
bOne = 0.1

epsilon = 0.1
delta = 0.01

def Dihotomy(a, b, epsilon, delta):
    c = a
    d = b

    while (d-c >= epsilon):
        x = (d + c - delta) / 2

        if funcOne(x) <= funcOne(x + delta):
            d = x + delta
        else:
            c = x;

    return (c,d)

def Fibbonachi(a,b, epsilon, delta):
    c = a
    d = b

    n = 2
    F = [1,1,2]

    while ((b-a)/F[n]) + (F[n-2]/F[n]) * delta >= epsilon:
        F.append(F[n] + F[n-1])
        n+=1

    x1 = b - ((F[n-1]/F[n])*(b-a) + ((-1)**n/F[n])*delta)
    x2 = a + b - x1

    for i in range(2, n):
        if funcOne(x1) < funcOne(x2):
            d = x2
            x2 = x1
            x1 = c + d - x2
        else:
            c = x1
            x1 = x2
            x2 = c + d - x1

    return (x1,x2)

for i in range(1,6):
    print("Dihotomy  for epsilon = {} : {}".format(epsilon**i, Dihotomy(aOne, bOne, epsilon**i, delta**i)))
    print("Fibbonachi  for epsilon = {} : {}".format(epsilon ** i, Fibbonachi(aOne, bOne, epsilon ** i, delta ** i)))
    print("Function value for epsilon = {} : {}".format(epsilon ** i, funcOne(Dihotomy(aOne,bOne,epsilon, delta)[0])))
    print("-"*20)


x = np.linspace(aOne, bOne, 100)

plt.plot(x, [funcOne(x) for x in x], 'g')  # график точного решения

xOpt = Dihotomy(aOne,bOne,epsilon, delta)[0]
yOpt = funcOne(xOpt)

plt.plot(xOpt,yOpt, 'ro')

plt.show()

