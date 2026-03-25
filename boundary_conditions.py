import numpy as np

#reset boundaries
def boundary_velocity(field):
    field[0, :] = 0
    field[:, 0] = 0
    field[-1, :] = 0
    field[:, -1] = 0
    
    return field

def boundary_pressure(p):
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]
    p[:, 0] = p[:, 1]
    p[:, -1] = p[:, -2]

    return p