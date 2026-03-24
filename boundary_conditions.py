import numpy as np

#reset boundaries
def boundary_check(u):
    u[0, :] = 0
    u[:, 0] = 0
    u[-1, :] = 0
    u[:, -1] = 0
    
    return u