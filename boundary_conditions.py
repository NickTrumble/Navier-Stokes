import numpy as np

#reset boundaries
def boundary_check(field):
    field[0, :] = 0
    field[:, 0] = 0
    field[-1, :] = 0
    field[:, -1] = 0
    
    return field