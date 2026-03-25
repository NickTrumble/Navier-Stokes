import numpy as np
import math
from boundary_conditions import *

def start_simulation(u, v, p, visc = 0.1):
    timestep = get_timestep(u, v, visc) #placeholder (replace with get_timestep(...))


    (u_star, v_star) = apply_advection(u, v, timestep)
    (u_diff, v_diff) = apply_diffusion(u_star, v_star, timestep, visc)
    (u_proj, v_proj, p) = apply_pressure_projection(u_diff, v_diff, p, timestep)

    #boundary
    u_proj = boundary_velocity(u_proj)
    v_proj = boundary_velocity(v_proj)

    #print(np.max(get_divergence(u_proj, v_proj)))
    return (u_proj, v_proj, p)

def sim_dye(dye, u, v, timestep, visc):
    size = dye.shape[0]
    dx = 1 / (size - 1)

    y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    x_prev = x - u * timestep / dx
    y_prev = y - v * timestep / dx

    x_prev = np.clip(x_prev, 0, size - 2)
    y_prev = np.clip(y_prev, 0, size - 2)

    xi = np.floor(x_prev).astype(int)
    yi = np.floor(y_prev).astype(int)

    xf = x_prev - xi
    yf = y_prev - yi

    dye_top = (1 - xf) * dye[yi, xi] + xf * dye[yi, xi + 1]
    dye_bottom = (1 - xf) * dye[yi + 1, xi] + xf * dye[yi + 1, xi + 1]
    return diffuse_dye((1 - yf) * dye_top + yf * dye_bottom, timestep, visc)

def diffuse_dye(dye, timestep, visc):
    return dye + get_laplacian(dye) * timestep * visc


#returns array of previous velocities
def apply_advection(u, v, timestep):
    size = u.shape[0] #symmetrical grid
    dx = 1 / (size - 1) 

    y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    x_prev = x - u * timestep / dx
    y_prev = y - v * timestep / dx

    x_prev = np.clip(x_prev, 0, size - 2)
    y_prev = np.clip(y_prev, 0, size - 2)

    return get_interpolated_vel(u, v, x_prev, y_prev)

#returns tuple of interpolated velocities for u and y
def get_interpolated_vel(u, v, x_prev, y_prev):
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

        p_star[1:-1, 1:-1] = (
            p[2:, 1:-1] +
            p[:-2, 1:-1] +
            p[1:-1, 2:] +
            p[1:-1, :-2] -
            dx * dx * div_star[1:-1, 1:-1]
        ) / 4

        #add boundary conditions
        p = boundary_pressure(p_star)

        if np.max(np.abs(p_star - p)) < 1e-6:
            break

    return p

def get_divergence(u, v):
    size = u.shape[0]#assume symmetrical
    dx = 1 / (size - 1)
    #divergence of u = du/dx + du/dy
    diver = np.zeros((size, size))

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

