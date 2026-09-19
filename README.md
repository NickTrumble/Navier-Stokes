# 2D Navier--Stokes Experiment

A Python/NumPy experiment for visualising a 2D velocity field, pressure field, and advected dye. It uses Matplotlib for interaction and display.

The update sequence applies advection, diffusion, pressure projection, velocity boundary conditions, and dye advection. The display can switch between dye, velocity vectors, and velocity magnitude.

## Running

```text
pip install numpy matplotlib
python main.py
```

Move the mouse with the left button held to add force; use the right mouse button to add dye. Press `1`, `2`, or `3` to select dye, vectors, or velocity magnitude.

## Status

This is a numerical learning project. It has not been validated against a reference CFD implementation, so its output should be treated as an experiment rather than engineering-grade fluid simulation.
