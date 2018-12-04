import numpy as np


def sign(x):
    return -1 if x < 0 else 1


def unit_vec(size, index):
    e = np.zeros((size, 1))
    e[index] = 1
    return e


def householder(a):
    m, n = a.shape
    v_s = []
    r = np.copy(a)
    for k in range(n):
        x = r[k:m, k:k+1]#
        v_k = sign(x[0,0]) * np.linalg.norm(x) * unit_vec(m - k, 0) + x
        v_k = v_k / np.linalg.norm(v_k)
        p = r[k:m, k:n]
        r[k:m, k:n] = p - 2 * v_k @ v_k.conj().T @ p
        v_s.append(v_k)
    r[np.isclose(r, 0)] = 0
    return v_s, r


def multiply_by_h(w, v):
    v_a = v.conj().T
    return w - 2 * (v_a @ w) / (v_a @ v) * v


# this does not work correctly#

#def compute_q(v_s):
#    m, n = len(v_s[0]), len(v_s)
#    q_a = np.diag(np.ones((m)))
#    for k in range(n - 1, -1, -1):
#        for l in range(m):
#            v_padded = np.zeros((m, 1))
#            v_padded[k:m, :] = v_s[k]
#            q_a[0:m, l:l+1] = multiply_by_h(q_a[0:m, l:l+1], v_padded)
#    return q_a.conj().T


a = np.random.sample((4, 3))
v_s, r = householder(a)
print(v_s)
print(r)

#q = compute_q(v_s)
#print(q)
#
#print(a)
#print(q @ r)
