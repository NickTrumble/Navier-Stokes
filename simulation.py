import numpy as np

def start_simulation(u, v):
    timestep = 0.001 #placeholder (replace with get_timestep(...))

    (u_star, v_star) = apply_advection(u, v, timestep)
    (u_diff, v_diff) = apply_diffusion(u_star, v_star, timestep)

       
    return

#returns array of previous velocities
def apply_advection(u, v, timestep):
    size = np.size(u, 0) #symmetrical grid

    u_prev = np.zeros((size, size))
    v_prev = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x_prev = i - u[i, j] * timestep
            y_prev = j - v[i, j] * timestep

            (u_star, v_star) = get_interpolated_vel(u, v, x_prev, y_prev)
            u_prev[i, j] = u_star
            v_prev[i, j] = v_star

    return (u_prev, v_prev)

#returns tuple of interpolated velocities for u and y
def get_interpolated_vel(u, v, x_prev, y_prev):
    #integer parts
    xi = np.floor(x_prev) 
    yi = np.floor(y_prev)

    #floating point parts
    xf = x_prev - xi
    yf = y_prev - yi

    #interpolate between left and right
    u_top = xf * u[xi, yi] + (1 - xf) * u[xi + 1, yi]
    v_top = xf * v[xi, yi] + (1 - xf) * v[xi + 1, yi]

    u_bottom = xf * u[xi, yi + 1] + (1 - xf) * u[xi + 1, yi + 1]
    v_bottom = xf * v[xi, yi + 1] + (1 - xf) * v[xi + 1, yi + 1]

    #interpolate between top and bottom
    u_star = yf * u_top + (1 - yf) * u_bottom
    v_star = yf * v_top + (1 - yf) * v_bottom

    return (u_star, v_star)

def apply_diffusion(u, v, timestep):
    #advected vel + discrete laplacian * timestep
    #discrete laplacian = sum of second order spatial partials  
    lap_u = get_laplacian(u)
    lap_v = get_laplacian(v)

    u += lap_u * timestep
    v += lap_v * timestep

    return (u, v)

#gets discrete laplacian for certain velocity field
def get_laplacian(field):
    size = np.size(field, 0)
    dx = 1 / size #assuming symmetrical

    laplacian = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            laplacian[i, j] = (
                field[i + 1, j] +
                field[i - 1, j] + 
                field[i, j + 1] +
                field[i, j - 1]
            ) / (dx * dx)
    return laplacian

def apply_pressure_projection(u, v, timestep):
    
    def_iter = 100

    diver = get_divergence(u, v)
    

def get_divergence(u, v):
    size = np.size(u, 0)#assume symmetrical
    dx = 1 / size
    #divergence of u = du/dx + du/dy
    diver = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            diver[i, j] = (
                u[i - 1, j] +
                u[i + 1, j] + 
                v[i, j - 1] + 
                v[i, j + 1]
            ) / (2 * dx)
    return diver

#gets dynamic timestep
def get_timestep(u, v, visc):
    C = 0.5 #safety

    dx = 1 / np.size(u, 0)
    dy = 1 / np.size(u, 1)

    u_max = np.max(np.abs(u)) #x component
    v_max = np.max(np.abs(v)) #y component

    off = 1e-5#for dividing by zero
    advection = np.min(dx / (u_max + off), dy / (v_max + off))
    diffusion = np.min(advection, dx ** 2 / (4 * visc))

    return C * diffusion