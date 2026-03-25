import matplotlib.pyplot as plt
import numpy as np
from grid import create_grid, create_random
from simulation import start_simulation

#first, generate environment

#define global variables
SIZE = 50 #50x50


#velocity fields
(u, v, p) = create_random(SIZE) #x-velocity, y-velocity, pressure

def quiver_plot():
    (X, Y) = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
    plt.quiver(X, Y, u, v)
    plt.title('Velocity field')
    plt.show()

def imshow_plot():
    speed = np.sqrt(u ** 2 + v ** 2)
    plt.imshow(speed, cmap='inferno')
    plt.colorbar()
    plt.show()

def plot_pressure():
    plt.imshow(p, cmap='inferno')
    plt.colorbar()
    plt.title('Pressure')
    plt.show()

def sim(u, v, p):
    plt.ion()
    fig, ax = plt.subplots()
    for i in range(1000):
        (u, v, p) = start_simulation(u, v, p)
        
        ax.clear()

        # speed = np.sqrt(u ** 2 + v ** 2)
        # im = ax.imshow(speed, cmap='inferno')
        # ax.set_title(f'Step: {i}')

        (X, Y) = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
        im = ax.quiver(X, Y, u, v)
        ax.set_title(f'Velocity field, step: {i}')

        plt.pause(0.001)
    plt.ioff()
    plt.show()

sim(u, v, p)
#quiver_plot()
#imshow_plot()
#plot_pressure()