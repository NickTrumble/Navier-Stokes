import numpy as np

#velocity fields (for arrow vectors)
#pressure fields
def create_grid(size):
    u = np.zeros((size, size))
    v = np.zeros((size, size))
    p = np.zeros((size, size))
    return (u, v, p)

#testing
def create_random(size):
    u = np.random.random((size, size))
    v = np.random.random((size, size))
    p = np.random.random((size, size))
    return (u, v, p)