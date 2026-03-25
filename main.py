import matplotlib.pyplot as plt
import numpy as np
from grid import *
from simulation import start_simulation, sim_dye, get_timestep

#define global variables
SIZE = 50 #50x50
visc = 0.1
(u, v, p) = create_random(SIZE) #x-velocity, y-velocity, pressure
dye = create_dye(SIZE)
timestep = get_timestep(u, v, visc)

#dye or vectors or fluids
mode = "dye" 

def add_force(u, v, x, y, radius = 3, strength = 0.5):
    Y, X = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing='ij')
    dist = np.sqrt((X - x) ** 2 + (Y- y) ** 2)
    mask = dist < radius

    dx = X - x
    dy = Y - y

    u[mask] += strength * dx[mask]
    v[mask] += strength * dy[mask]

def on_move(event):
    global u, v, dye
    if event.button == 1 and event.xdata and event.ydata:
        x = int(event.xdata)
        y = int(event.ydata)
        add_force(u, v, x, y)
    if event.button == 3 and event.xdata and event.ydata:
        x = int(event.xdata)
        y = int(event.ydata)
        add_dye(dye, x, y)

def add_dye(dye, x, y, radius = 2, amount = 0.1):
    Y, X = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing='ij')
    dist = np.sqrt((X - x) ** 2 + (Y- y) ** 2)
    mask = dist < radius

    falloff = (radius - dist[mask]) / radius

    dye[mask] += amount * falloff

def on_key(event):
    global mode
    if event.key == '1':
        mode = "dye"
    if event.key == '2':
        mode = "vectors"
    if event.key == '3':
        mode = "fluids"

fig, ax = plt.subplots()
(X, Y) = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
quiver = ax.quiver(X, Y, u, v)
speed = np.sqrt(u[1:-1, 1:-1] ** 2 + v[1:-1, 1:-1] ** 2)
im = ax.imshow(dye, cmap='Greys')
plt.colorbar(im)

def update_plot(i):
    global mode
    if mode == "dye":
        quiver.set_visible(False)
        ax.set_title(f'Dye, step: {i}')
        im.set_data(dye)
        im.set_visible(True)
    elif mode == "vectors":
        im.set_visible(False)
        quiver.set_UVC(u, v)
        ax.set_title(f'Velocity field, step: {i}')
        quiver.set_visible(True)
    else:
        quiver.set_visible(False)
        im.set_visible(True)
        speed = np.sqrt(u[1:-1, 1:-1] ** 2 + v[1:-1, 1:-1] ** 2)
        im.set_data(speed)
        ax.set_title(f'Velocity heat map, step: {i}')

def sim():
    global u, v, p, dye
    plt.ion()

    for i in range(1000):
        (u, v, p) = start_simulation(u, v, p)
        dye = sim_dye(dye, u, v, timestep, visc)
        update_plot(i)

        plt.pause(0.01)

    plt.ioff()
    plt.show()



fig.canvas.mpl_connect("motion_notify_event", on_move)
fig.canvas.mpl_connect("key_press_event", on_key)
sim()
#quiver_plot()
#imshow_plot()
#plot_pressure()