import numpy as np

# a)
# fertig

# b)
print('b)', np.zeros((1, 4)))

# c)
print('c)', np.ones((5, 1)))

# d)
# numpy documentation recommends not to use the matrix class, but to use arrays instead
print('d)', np.zeros((2,2)))

# e)
mat = np.arange(1, 13).reshape((4, 3))
print('e)', mat[1, :])

# f)
print('f)', mat[:, 2])

# g)
print('g)', mat.T)

# h)
mat2 = np.arange(1, 10).reshape((3, 3))
mat3 = np.arange(1, 10).reshape((3, 3))
print('h)', mat2, np.matmul(mat2, mat3), np.multiply(mat2, mat3), sep='\n')

# i)
print('i)', np.hstack((mat2, mat3)), np.vstack((mat2, mat3)), sep='\n')

# j)
print('j)', mat.shape)

# k)
print('k)', np.arange(1, 57).reshape(8, 7).reshape(14,4))

# l)
print('l)', np.arange(1, 4).reshape((3, 1)).repeat(10000, 1))

# m)
mat2[mat2 < 0] = 0

# n)
np.arange(1, 100, 7)

# o)
vec = np.ones(100)
vec[np.arange(0,100) % 2 == 0] = 0
print('o)', vec)

# p)
vec2 = np.arange(100)
np.delete(vec2, np.arange(0, 100, 2))

# q)

# r)


def det(arr):
    return arr[0] * arr[3] - arr[1] * arr[2]


def invert(arr):
    d = det(arr)
    if d == 0:
        arr[:] = 0
    else:
        d1 = det(np.concatenate(([0.], arr[1:])))
        d2 = det(np.concatenate((arr[:1],  [0.], arr[2:])))
        d3 = det(np.concatenate((arr[:2], [0.],arr[3:])))
        d4 = det(np.concatenate((arr[:3], [0.])))
        arr[0] = d1 / d
        arr[1] = d2 / d
        arr[2] = d3 / d
        arr[3] = d4 / d


mat = np.random.random((1000, 4))

mat2 = np.copy(mat)

print('r) before:', mat, sep='\n')

np.apply_along_axis(invert, 1, mat)

print('after:', mat, sep='\n')

list(map(invert, mat2))

print('after:', mat2, sep='\n')
