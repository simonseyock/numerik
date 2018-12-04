from numpy import *
from matplotlib import pyplot

x = arange(0.0, 2.0*pi,0.01)
x = x.reshape((x.size, 1))
y = sin(x)

x0 = 0.5 * pi
sigma = 0.5
w = exp(-(x - x0) ** 2 / (2.0 * sigma ** 2))

A = matrix(hstack([x**0, x**1, x**2]))
W = matrix(diagflat(sqrt(w)))

p = linalg.lstsq(W * A, W * y)[0]

pyplot.plot(x, y)
pyplot.plot(x, (A * p).getA())
pyplot.show()
