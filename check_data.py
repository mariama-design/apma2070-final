"""Step 0 sanity check: load the spectral reference, interpolate to a 201x201 grid,
plot u, v, p and report a few basic numbers. The figure should look like the project's
Figure 2.
"""

import numpy as np
import matplotlib.pyplot as plt
import utils

xi, yi, U, V, P = utils.load_reference_grid(n=201)
dx, dy = utils.grid_spacing(xi, yi)

print(f"grid: {len(xi)} x {len(yi)}, dx = {dx:.4g}")
print(f"u range: [{U.min():.4f}, {U.max():.4f}]")
print(f"v range: [{V.min():.4f}, {V.max():.4f}]")
print(f"p range: [{P.min():.4f}, {P.max():.4f}]")
print(f"||div u||_L2 (reference, FD on 201x201): {utils.divergence_l2(U, V, dx, dy):.4e}")
print(f"max |vorticity|: {np.abs(utils.vorticity(U, V, dx, dy)).max():.4f}")

# Centerlines for later comparison with PINN runs.
mid = len(xi) // 2
u_centerline_vertical = U[:, mid]   # u(x=0.5, y)
v_centerline_horizontal = V[mid, :] # v(x, y=0.5)
np.savez(
    "reference_grid.npz",
    xi=xi, yi=yi, U=U, V=V, P=P,
    u_x05_vs_y=u_centerline_vertical,
    v_y05_vs_x=v_centerline_horizontal,
)
print("saved reference_grid.npz")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, F, name, cmap in zip(axes, [U, V, P], ["u", "v", "p"], ["jet", "jet", "jet"]):
    cf = ax.contourf(xi, yi, F, levels=30, cmap=cmap)
    fig.colorbar(cf, ax=ax)
    ax.set_title(f"{name}(x, y) — spectral reference, Re = 100")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("check_data.png", dpi=150)
print("saved check_data.png")
