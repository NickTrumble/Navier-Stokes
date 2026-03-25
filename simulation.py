import numpy as np
import math
from boundary_conditions import *

def start_simulation(u, v, p):
    visc = 0.1
    timestep = get_timestep(u, v, visc) #placeholder (replace with get_timestep(...))


    (u_star, v_star) = apply_advection(u, v, timestep)
    (u_diff, v_diff) = apply_diffusion(u_star, v_star, timestep, visc)
    (u_proj, v_proj, p) = apply_pressure_projection(u_diff, v_diff, p, timestep)

    #boundary
    # u_proj = boundary_check(u_proj)
    # v_proj = boundary_check(v_proj)

    print(np.max(get_divergence(u_proj, v_proj)))
    return (u_proj, v_proj, p)

#returns array of previous velocities
def apply_advection(u, v, timestep):
    size = u.shape[0] #symmetrical grid
    dx = 1 / (size - 1) 

    # u_prev = np.zeros((size, size))
    # v_prev = np.zeros((size, size))
    # for i in range(size):
    #     for j in range(size):
    #         x_prev = max(0, min(size - 2, i - u[i, j] * timestep / dx))
    #         y_prev = max(0, min(size - 2, j - v[i, j] * timestep / dx))

    #         (u_star, v_star) = get_interpolated_vel(u, v, x_prev, y_prev)
    #         u_prev[i, j] = u_star
    #         v_prev[i, j] = v_star

    # return (u_prev, v_prev)

    y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    x_prev = x - u * timestep / dx
    y_prev = y - v * timestep / dx

    x_prev = np.clip(x_prev, 0, size - 2)
    y_prev = np.clip(y_prev, 0, size - 2)

    return get_interpolated_vel(u, v, x_prev, y_prev)

#returns tuple of interpolated velocities for u and y
def get_interpolated_vel(u, v, x_prev, y_prev):
    # #integer parts
    # xi = math.floor(x_prev) 
    # yi = math.floor(y_prev)

    # #floating point parts
    # xf = x_prev - xi
    # yf = y_prev - yi

    # #interpolate between left and right
    # u_top = (1 - xf) * u[xi, yi] + xf * u[xi + 1, yi]
    # v_top = (1 - xf) * v[xi, yi] + xf * v[xi + 1, yi]

    # u_bottom = (1 - xf) * u[xi, yi + 1] + xf * u[xi + 1, yi + 1]
    # v_bottom = (1 - xf) * v[xi, yi + 1] + xf * v[xi + 1, yi + 1]

    # #interpolate between top and bottom
    # u_star = (1 - yf) * u_top + yf * u_bottom
    # v_star = (1 - yf) * v_top + yf * v_bottom

    
    xi = np.floor(x_prev).astype(int)
    yi = np.floor(y_prev).astype(int)

    xf = x_prev - xi
    yf = y_prev - yi

    u_top = (1 - xf) * u[yi, xi] + xf * u[yi, xi + 1]
    v_top = (1 - xf) * v[yi, xi] + xf * v[yi, xi + 1]

    u_bottom = (1 - xf) * u[yi + 1, xi] + xf * u[yi + 1, xi + 1]
    v_bottom = (1 - xf) * v[yi + 1, xi] + xf * v[yi + 1, xi + 1]

    #interpolate between top and bottom
    u_star = (1 - yf) * u_top + yf * u_bottom
    v_star = (1 - yf) * v_top + yf * v_bottom


    return (u_star, v_star)

def apply_diffusion(u, v, timestep, visc):
    #advected vel + discrete laplacian * timestep
    #discrete laplacian = sum of second order spatial partials  
    lap_u = get_laplacian(u)
    lap_v = get_laplacian(v)

    u_star = u + lap_u * timestep * visc
    v_star = v + lap_v * timestep * visc

    return (u_star, v_star)

#gets discrete laplacian for certain velocity field
def get_laplacian(field):
    size = np.size(field, 0)
    dx = 1 / (size - 1) #assuming symmetrical

    laplacian = np.zeros((size, size))
    # for i in range(1, size - 1):
    #     for j in range(1, size - 1):
    #         laplacian[i, j] = (
    #             field[i + 1, j] +
    #             field[i - 1, j] + 
    #             field[i, j + 1] +
    #             field[i, j - 1] -
    #             4 * field[i, j]
    #         ) / (dx * dx)

    laplacian[1:-1, 1:-1] = (
        field[2:, 1:-1] +
        field[:-2, 1:-1] +
        field[1:-1, 2:] +
        field[1:-1, :-2] -
        4 * field[1:-1, 1:-1]
    ) / (dx * dx)
    return laplacian

def apply_pressure_projection(u, v, p, timestep):
    size = p.shape[0] #assuming symmtrical

    diver = get_divergence(u, v)
    p = iterate_jacobi(size, p, diver, timestep)

    (dpdx, dpdy) = get_grad_p(p, size)

    u_star = u - dpdx * timestep
    v_star = v - dpdy * timestep

    return (u_star, v_star, p)

def get_grad_p(p, size):
    dx = 1 / (size - 1)

    dpdx = np.zeros((size, size))
    dpdy = np.zeros((size, size))

    # for i in range(1, size - 1):
    #     for j in range(1, size - 1):
    #         dpdx[i, j] = (
    #             p[i + 1, j] -
    #             p[i - 1, j]
    #         ) / (2 * dx)
    #         dpdy[i, j] = (
    #             p[i, j + 1] -
    #             p[i, j - 1]
    #         ) / (2 * dx)

    dpdx[1:-1, 1:-1] = (
        p[2:, 1:-1] -
        p[:-2, 1:-1]
    ) / (2 * dx)
    dpdy[1:-1, 1:-1] = (
        p[1:-1, 2:] -
        p[1:-1, :-2]
    ) / (2 * dx)

    return (dpdx, dpdy)

def iterate_jacobi(size, p, diver, timestep, def_iter=500):
    dx = 1 / (size - 1)
    div_star = diver / timestep
    for _ in range(def_iter):
        p_star = p.copy()

        # for i in range(1, size - 1):
        #     for j in range(1, size - 1):
        #         p_star[i, j] = (
        #             p[i + 1, j] +
        #             p[i - 1, j] +
        #             p[i, j + 1] +
        #             p[i, j - 1] -
        #             dx * dx * div_star[i, j]
        #         ) / 4
        
        p_star[1:-1, 1:-1] = (
            p[2:, 1:-1] +
            p[:-2, 1:-1] +
            p[1:-1, 2:] +
            p[1:-1, :-2] -
            dx * dx * div_star[1:-1, 1:-1]
        ) / 4

        #add boundary conditions
        p = boundary_pressure(p_star)
    return p

def get_divergence(u, v):
    size = u.shape[0]#assume symmetrical
    dx = 1 / (size - 1)
    #divergence of u = du/dx + du/dy
    diver = np.zeros((size, size))

    # for i in range(1, size - 1):
    #     for j in range(1, size - 1):
    #         diver[i, j] = (
    #             u[i + 1, j] -
    #             u[i - 1, j] + 
    #             v[i, j + 1] -
    #             v[i, j - 1]
    #         ) / (2 * dx)

    diver[1:-1, 1:-1] = (
        u[2:, 1:-1] -
        u[:-2, 1:-1] +
        v[1:-1, 2:] -
        v[1:-1, :-2]
    ) / (2 * dx)

    return diver

#gets dynamic timestep
def get_timestep(u, v, visc):
    C = 0.5 #safety

    dx = 1 / (u.shape[0] - 1)
    dy = 1 / (u.shape[0] - 1)

    u_max = np.max(np.abs(u)) #x component
    v_max = np.max(np.abs(v)) #y component

    off = 1e-5#for dividing by zero
    advection = min(dx / (u_max + off), dy / (v_max + off))
    diffusion = min(advection, dx ** 2 / (4 * visc))

    return C * diffusion