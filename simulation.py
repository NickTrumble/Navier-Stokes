import numpy as np

def start_simulation():
    timestep = 0.001 #(placeholder)



    return

def get_timestep(u, v):
    C = 0.5 #safety

    dx = 1 / np.array(u, 0)
    dy = 1 / np.array(u, 1)

    u_max = np.max(np.abs(u)) #x component
    v_max = np.max(np.abs(v)) #y component

    off = 1e-5#for dividing by zero
    advection = np.min(dx / (u_max + off), dy / (v_max + off))