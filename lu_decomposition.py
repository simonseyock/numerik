import numpy as np

def lu_decomposition(a):
    m = a.shape[0]
    u = np.copy(a)
    l = np.diag(np.ones(m))

    for k in range(m - 1):
        for j in range(k + 1, m):
            l[j, k] = u[j, k] / u[k, k]
            u[j, k:m] = u[j, k:m] - l[j, k] * u[k, k:m]

    u[np.isclose(u, 0)] = 0
    return l, u


def back_substitution(u, b):
    n = a.shape[1]
    b_copy = np.copy(b)
    x = np.zeros(b.shape)
    for i in range(n - 1, -1, -1):
        x[i] = b_copy[i] / u[i, i]
        b[0: i-1] = b[0: i-1] - u[0: i-1, i: i+1] * x[i]
    return x


def forward_substitution(l, b):
    n = a.shape[1]
    b_copy = np.copy(b)
    x = np.zeros(b.shape)
    for i in range(n):
        x[i] = b_copy[i] / l[i, i]
        b[i+1: n-1] = b[i+1: n-1] - l[i+1: n-1, i: i+1] * x[i]
    return x


a = np.random.sample((5, 5))
b = np.random.sample((5, 1))

print(a)
l, u = lu_decomposition(a)
print(l)
print(u)
print(l @ u)

print(back_substitution(u, b))
print(forward_substitution(l, b))