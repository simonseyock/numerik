import numpy as np


def calculate_svd(mat_a: np.ndarray):
    b = mat_a.conj().T @ mat_a
    eig_values, eig_vectors = np.linalg.eig(b)
    sigma = np.diag(np.sqrt(eig_values))
    # normally in the case that |leigen_values != 0| < m
    # you would add vectors to u till it is an orthonormal base
    # here we just set them to 0
    u = mat_a @ eig_vectors * np.divide(1, np.sqrt(eig_values), where=(eig_values != 0), out=np.zeros_like(eig_values))
    return u, sigma, eig_vectors.conj().T


def pseudo_inverse(mat_a: np.ndarray):
    u, sigma, v_h = calculate_svd(mat_a)
    sigma_plus = np.divide(1, sigma, where=(sigma != 0), out=np.zeros_like(sigma))
    return v_h.conj().T @ sigma_plus @ u.conj().T


def linear_solve(mat_a: np.ndarray, vec_b: np.ndarray):
    return pseudo_inverse(mat_a) @ vec_b


def set_zeros(a):
    a[np.isclose(r, np.zeros(a.shape))] = 0

# # test for svd
# test = np.random.random((4,5))
# u, sigma, v_h = calculate_svd(test)
# print(test)
# print(u @ sigma @ v_h)

# test for linear_solve

mat_a = np.array([[3, 2, -1], [2, -2, 4], [-1, 1/2, -1]], dtype=np.float64)
vec_b = np.array([[1], [-2], [0]], dtype=np.float64)

print('test svd')

print('A', mat_a)

u, sigma, v_h = calculate_svd(mat_a)

print('U', u)
print(u'\u03A3', sigma)
print('V*', v_h)

print(u'U \u03A3 V*', u @ sigma @ v_h)

print('\n\ntest pseudo_inverse')

print('A', mat_a)
mat_a_i = pseudo_inverse(mat_a)
print('A+', mat_a_i)
r = mat_a @ mat_a_i
print('A A+', r)
print('set 0 where close to 0')
set_zeros(r)
print('A A+', r)

print('\n\ntest linear_solve')

print('A', mat_a)
print('b', vec_b)

sol = linear_solve(mat_a, vec_b)
print('solution ', sol)

r = mat_a @ sol
print('Ax', r)
print('set 0 where close to 0')
set_zeros(r)
print('Ax', r)