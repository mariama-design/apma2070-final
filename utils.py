"""Small helpers used across tasks: data loading, error metrics, finite-difference diagnostics."""

import os
import numpy as np
import scipy.io
from scipy.interpolate import griddata

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_cavity_data():
    """Return (x, y, u, v, p) as 1D arrays of length 20201 (scattered spectral nodes)."""
    vel = scipy.io.loadmat(os.path.join(DATA_DIR, "velocity.mat"))
    pre = scipy.io.loadmat(os.path.join(DATA_DIR, "pressure.mat"))
    x = vel["x"].ravel()
    y = vel["y"].ravel()
    u = vel["u"].ravel()
    v = vel["v"].ravel()
    p = pre["p"].ravel()
    return x, y, u, v, p


def to_uniform_grid(x, y, f, n=201):
    """Interpolate scattered (x,y,f) onto a uniform n x n grid on [0,1]^2.

    Returns (xi, yi, F) where F has shape (n, n), F[i, j] is the value at (xi[j], yi[i]).
    """
    xi = np.linspace(0.0, 1.0, n)
    yi = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(xi, yi)
    F = griddata((x, y), f, (X, Y), method="linear")
    # Boundary points can fall outside the convex hull of the scattered set; fill with nearest.
    nan = np.isnan(F)
    if nan.any():
        F[nan] = griddata((x, y), f, (X[nan], Y[nan]), method="nearest")
    return xi, yi, F


def load_reference_grid(n=201):
    """Load and interpolate u, v, p onto a uniform n x n grid. Returns (xi, yi, U, V, P)."""
    x, y, u, v, p = load_cavity_data()
    xi, yi, U = to_uniform_grid(x, y, u, n=n)
    _, _, V = to_uniform_grid(x, y, v, n=n)
    _, _, P = to_uniform_grid(x, y, p, n=n)
    return xi, yi, U, V, P


def relative_l2(pred, ref):
    """Relative L^2 error ||pred - ref||_2 / ||ref||_2 over all grid points."""
    pred = np.asarray(pred).ravel()
    ref = np.asarray(ref).ravel()
    return np.linalg.norm(pred - ref) / np.linalg.norm(ref)


def grid_spacing(xi, yi):
    """Uniform grid spacing (dx, dy)."""
    return xi[1] - xi[0], yi[1] - yi[0]


def vorticity(U, V, dx, dy):
    """omega = dv/dx - du/dy on a uniform grid. Central differences via np.gradient."""
    dVdx = np.gradient(V, dx, axis=1)
    dUdy = np.gradient(U, dy, axis=0)
    return dVdx - dUdy


def divergence(U, V, dx, dy):
    """div u = du/dx + dv/dy on a uniform grid."""
    dUdx = np.gradient(U, dx, axis=1)
    dVdy = np.gradient(V, dy, axis=0)
    return dUdx + dVdy


def divergence_l2(U, V, dx, dy):
    """L^2 norm of the divergence field over the unit square."""
    div = divergence(U, V, dx, dy)
    return np.sqrt(np.mean(div ** 2))
