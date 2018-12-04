import numpy as np

# calculate a_plus

a = np.array([[1, -1, 1], [1, 0, 0], [1, 1, 1], [1, 2, 4]], dtype=np.float64)
b = np.array([1, 0, 2, 4], dtype=np.float64).reshape((4,1))
u, sigma, v_t = np.linalg.svd(a)
tmp = np.append(np.divide(1, sigma, where=(sigma != 0)), np.zeros(a.shape[0] - sigma.shape[0]))
sigma_plus = np.resize(np.diag(tmp), (a.shape[1], a.shape[0]))


a_plus = v_t.T @ np.diag(sigma_plus) @ u.T