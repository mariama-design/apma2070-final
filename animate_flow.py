"""Animate tracer particles in the steady Re=100 cavity velocity field.

Background: speed magnitude + streamlines (the flow doesn't change in time, the
particles just move through it). RK4 integration with a fixed dt, particles that
hit a wall get reseeded so the animation stays full.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import RegularGridInterpolator

ref = np.load("reference_grid.npz")
xi, yi, U, V = ref["xi"], ref["yi"], ref["U"], ref["V"]

# RegularGridInterpolator expects (y, x) ordering since U has shape (len(yi), len(xi)).
u_interp = RegularGridInterpolator((yi, xi), U, bounds_error=False, fill_value=0.0)
v_interp = RegularGridInterpolator((yi, xi), V, bounds_error=False, fill_value=0.0)


def velocity(px, py):
    pts = np.column_stack([np.clip(py, 0.0, 1.0), np.clip(px, 0.0, 1.0)])
    return u_interp(pts), v_interp(pts)


def rk4_step(px, py, dt):
    k1u, k1v = velocity(px, py)
    k2u, k2v = velocity(px + 0.5 * dt * k1u, py + 0.5 * dt * k1v)
    k3u, k3v = velocity(px + 0.5 * dt * k2u, py + 0.5 * dt * k2v)
    k4u, k4v = velocity(px + dt * k3u, py + dt * k3v)
    px_new = px + (dt / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
    py_new = py + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return px_new, py_new


# Particle seeding.
N = 400
rng = np.random.default_rng(0)
px = rng.uniform(0.03, 0.97, N)
py = rng.uniform(0.03, 0.97, N)

dt = 0.01
n_frames = 240
fps = 24

fig, ax = plt.subplots(figsize=(7, 7))
speed = np.sqrt(U ** 2 + V ** 2)
ax.contourf(xi, yi, speed, levels=30, cmap="viridis")
ax.streamplot(xi, yi, U, V, density=1.4, color="white", linewidth=0.5, arrowsize=0.6)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Re = 100 lid-driven cavity — tracer particles in steady velocity field")

scat = ax.scatter(px, py, s=14, c="red", edgecolors="white", linewidths=0.4, zorder=10)


def update(frame):
    global px, py
    px, py = rk4_step(px, py, dt)
    # Reseed particles that ran into a wall (no-slip would stall them anyway).
    hit = (px < 0.005) | (px > 0.995) | (py < 0.005) | (py > 0.995)
    if hit.any():
        n = int(hit.sum())
        px[hit] = rng.uniform(0.05, 0.95, n)
        py[hit] = rng.uniform(0.05, 0.95, n)
    scat.set_offsets(np.column_stack([px, py]))
    return (scat,)


anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=True)
anim.save("cavity_flow.gif", writer=PillowWriter(fps=fps))
print("saved cavity_flow.gif")
