import numpy as np

#velocity fields (for arrow vectors)
#pressure fields
def create_grid(size):
    u = np.ones((size, size)) * 1.0
    v = np.zeros((size, size))
    p = np.random.random((size, size))
    return (u, v, p)

#testing
def create_random(size):
    u = np.random.random((size, size)) * 2 - 1
    v = np.random.random((size, size)) * 2 - 1
    p = np.zeros((size, size))
    return (u, v, p)