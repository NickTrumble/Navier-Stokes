import matplotlib.pyplot as plt
import numpy as np
from grid import create_grid, create_random


#first, generate environment

#define global variables
SIZE = 10 #50x50


#velocity fields
(u, v, p) = create_random(SIZE) #x-velocity, y-velocity, pressure

def quiver_plot():
    plt.quiver(u, v)
    plt.show()

def imshow_plot():
    speed = np.sqrt(u ** 2 + v ** 2)
    plt.imshow(speed, cmap='inferno')
    plt.colorbar()
    plt.show()

#quiver_plot()
imshow_plot()