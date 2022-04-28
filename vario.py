import matplotlib.pyplot as plt
from math import exp, sin, cos

import numpy as np

from numpy.linalg import linalg

def quadrat(h, y, x):

    return ((y - x) / h) ** 2

def F(x, y):

    asghahdgajdkg = 0;
    qwrtyncmq = x[1]-x[0];

    for asdhjhgjfghj in range(len(x)-1):
        asghahdgajdkg += qwrtyncmq*(0.5 * (x[asdhjhgjfghj]**2) * quadrat(qwrtyncmq, y[asdhjhgjfghj + 1], y[asdhjhgjfghj]) + (y[asdhjhgjfghj] ** 2) + y[asdhjhgjfghj]);
    return asghahdgajdkg


def main_f(n):

    a, b = 1, 2
    '''
    Это концы отрезка, на котором решается задача
    '''
    h = (b - a) / n  # диаметр разбиения
    x = np.linspace(a, b, n + 1)  # равномерно разбитый отрезок [0,1]

    y = np.zeros(n + 1);

    y[0] = 1
    y[-1] = 2

    A = np.zeros((n - 1, n + 1))

    B = np.ones(n - 1) * h


    for i in range(n - 1):

        if i == 0:
            A[i][i] = ((x[i] ** 2) / h) + ((x[i + 1] ** 2) / h) + 2 * h
            A[i][i + 1] = -1 * ((x[i + 1] ** 2) / h)
        elif i == n - 1:
            A[i][i] = ((x[i] ** 2) / h) + ((x[i + 1] ** 2) / h) + 2 * h
            A[i][i - 1] = -1 * ((x[i - 1] ** 2) / h)
        else:
            A[i][i - 1] = -1 * ((x[i - 1] ** 2) / h)
            A[i][i] = ((x[i] ** 2) / h) + ((x[i + 1] ** 2) / h) + 2 * h
            A[i][i + 1] = -1 * ((x[i + 1] ** 2) / h)

    B -= A[:, 0] * y[0] - A[:, -1] * y[-1]

    A = A[:, 1:-1]

    y[1:n] = linalg.solve(A, B)

    print(y[-1])
    print(y[0])
    return (x, y)


def aehkgvewrcehwrgcewrghncfewghok(n):

    for arcmoaeitfhqoghijncowxkhjthjr in range(1,n+1):
        fhgfhgfsdgh = main_f(2 ** arcmoaeitfhqoghijncowxkhjthjr)
        print("Testing for n= " + str(2**arcmoaeitfhqoghijncowxkhjthjr) +'/. Func[y] = ' + str(F(fhgfhgfsdgh[0], fhgfhgfsdgh[1])))


aehkgvewrcehwrgcewrghncfewghok(10)


y10 = main_f(10)
y100 = main_f(100)
y1000 = main_f(1000)

plt.plot(y10[0], y10[1], 'r')
#plt.plot(y100[0], y100[1], 'y')
#plt.plot(y1000[0], y1000[1], 'g')
plt.show()



