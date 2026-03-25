import matplotlib.pyplot as plt
import numpy as np
from grid import create_grid, create_random
from simulation import start_simulation

#first, generate environment

#define global variables
SIZE = 50 #50x50


#velocity fields
(u, v, p) = create_random(SIZE) #x-velocity, y-velocity, pressure

def add_force(u, v, x, y, radius = 3, strength = 0.5):
    Y, X = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing='ij')
    dist = np.sqrt((X - x) ** 2 + (Y- y) ** 2)
    mask = dist < radius

    dx = X - x
    dy = Y - y

    u[mask] += strength * dx[mask]
    v[mask] += strength * dy[mask]


def on_move(event):
    global u, v
    if event.button == 1 and event.xdata and event.ydata:
        x = int(event.xdata)
        y = int(event.ydata)
        add_force(u, v, x, y)


fig, ax = plt.subplots()
# (X, Y) = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
# quiver = ax.quiver(X, Y, u, v)
speed = np.sqrt(u[1:-1, 1:-1] ** 2 + v[1:-1, 1:-1] ** 2)
im = ax.imshow(speed, cmap='inferno')

def sim():
    global u, v, p
    plt.ion()

    for i in range(1000):
        (u, v, p) = start_simulation(u, v, p)
        speed = np.sqrt(u[1:-1, 1:-1] ** 2 + v[1:-1, 1:-1] ** 2)
        
        ax.set_title(f'Step: {i}')
        im.set_data(speed)
        # quiver.set_UVC(u, v)
        # ax.set_title(f'Velocity field, step: {i}')

        plt.pause(0.01)
    plt.ioff()
    plt.show()



fig.canvas.mpl_connect("motion_notify_event", on_move)
sim()
#quiver_plot()
#imshow_plot()
#plot_pressure()