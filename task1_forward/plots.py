"""Plot Task 1 results: contours, vorticity, centerlines, error heatmaps, loss curve.

Usage:
  python task1_forward/plots.py --tag baseline
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import utils  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, default="baseline")
args = parser.parse_args()

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
field = np.load(os.path.join(RESULTS, f"{args.tag}_field.npz"))
xi = field["xi"]; yi = field["yi"]
U_pred = field["U"]; V_pred = field["V"]; P_pred = field["P"]

ref = np.load(os.path.join(REPO, "reference_grid.npz"))
U_ref = ref["U"]; V_ref = ref["V"]; P_ref = ref["P"]

# Pressure is defined up to a constant. Zero-center both pred and ref before comparing.
P_pred_c = P_pred - P_pred.mean()
P_ref_c = P_ref - P_ref.mean()

dx, dy = utils.grid_spacing(xi, yi)
omega_pred = utils.vorticity(U_pred, V_pred, dx, dy)
omega_ref = utils.vorticity(U_ref, V_ref, dx, dy)
div_pred = utils.divergence(U_pred, V_pred, dx, dy)
div_pred_l2 = np.sqrt(np.mean(div_pred ** 2))

# Relative L2 errors.
err_u = utils.relative_l2(U_pred, U_ref)
err_v = utils.relative_l2(V_pred, V_ref)
err_p = utils.relative_l2(P_pred_c, P_ref_c)

print(f"=== {args.tag} ===")
print(f"rel-L2 u: {err_u:.4f}")
print(f"rel-L2 v: {err_v:.4f}")
print(f"rel-L2 p: {err_p:.4f}    (after zero-mean gauge)")
print(f"||div u||_L2 (FD on PINN field): {div_pred_l2:.4e}")
print(f"max |omega_pred|: {np.abs(omega_pred).max():.2f}")


# ---- Fig 1: u, v, p contours, pred vs reference ----
def clip_p(P, q=99):
    lo, hi = np.percentile(P, [100 - q, q])
    m = max(abs(lo), abs(hi))
    return -m, m

fig, axes = plt.subplots(3, 3, figsize=(13, 12))
fields = [
    ("u", U_pred, U_ref, "jet", None),
    ("v", V_pred, V_ref, "jet", None),
    ("p", P_pred_c, P_ref_c, "jet", "clip"),
]
for row, (name, F_pred, F_ref, cmap, clip) in enumerate(fields):
    if clip == "clip":
        vmin, vmax = clip_p(F_ref, 99)
    else:
        vmin = min(F_pred.min(), F_ref.min())
        vmax = max(F_pred.max(), F_ref.max())
    for col, (F, title) in enumerate([
        (F_pred, f"{name} — PINN ({args.tag})"),
        (F_ref, f"{name} — spectral reference"),
        (F_pred - F_ref, f"{name} error (PINN − ref)")
    ]):
        cmap_here = cmap if col < 2 else "RdBu_r"
        if col < 2:
            cf = axes[row, col].contourf(xi, yi, F, levels=30, cmap=cmap_here, vmin=vmin, vmax=vmax)
        else:
            m = np.abs(F).max()
            cf = axes[row, col].contourf(xi, yi, F, levels=30, cmap=cmap_here, vmin=-m, vmax=m)
        plt.colorbar(cf, ax=axes[row, col])
        axes[row, col].set_title(title)
        axes[row, col].set_aspect("equal")
        axes[row, col].set_xlabel("x"); axes[row, col].set_ylabel("y")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, f"{args.tag}_fields.png"), dpi=130)
plt.close()


# ---- Fig 2: vorticity & divergence ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
m = np.percentile(np.abs(omega_ref), 99)
for ax, F, title in zip(axes[:2],
                        [omega_pred, omega_ref],
                        [f"vorticity ω — PINN ({args.tag})", "vorticity ω — reference"]):
    cf = ax.contourf(xi, yi, F, levels=30, cmap="RdBu_r", vmin=-m, vmax=m)
    plt.colorbar(cf, ax=ax)
    ax.set_title(title); ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
cf = axes[2].contourf(xi, yi, np.log10(np.abs(div_pred) + 1e-10), levels=30, cmap="viridis")
plt.colorbar(cf, ax=axes[2])
axes[2].set_title(f"log10 |∇·u| — PINN ({args.tag})\n‖∇·u‖₂ = {div_pred_l2:.2e}")
axes[2].set_aspect("equal"); axes[2].set_xlabel("x"); axes[2].set_ylabel("y")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, f"{args.tag}_vorticity_div.png"), dpi=130)
plt.close()


# ---- Fig 3: centerline profiles ----
mid = len(xi) // 2
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(yi, U_ref[:, mid], "k-", lw=2, label="spectral (reference)")
axes[0].plot(yi, U_pred[:, mid], "r--", lw=1.5, label=f"PINN ({args.tag})")
axes[0].set_xlabel("y"); axes[0].set_ylabel("u")
axes[0].set_title(f"Centerline u(x=0.5, y)   rel-L² = {err_u:.4f}")
axes[0].grid(True, alpha=0.3); axes[0].legend()

axes[1].plot(xi, V_ref[mid, :], "k-", lw=2, label="spectral (reference)")
axes[1].plot(xi, V_pred[mid, :], "r--", lw=1.5, label=f"PINN ({args.tag})")
axes[1].set_xlabel("x"); axes[1].set_ylabel("v")
axes[1].set_title(f"Centerline v(x, y=0.5)   rel-L² = {err_v:.4f}")
axes[1].grid(True, alpha=0.3); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, f"{args.tag}_centerlines.png"), dpi=130)
plt.close()


# ---- Fig 4: loss curve ----
loss = np.load(os.path.join(RESULTS, f"{args.tag}_loss.npz"))
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(loss["iters"], loss["loss"], label="total", lw=2)
ax.semilogy(loss["iters"], loss["pde"], label="pde residual")
ax.semilogy(loss["iters"], loss["bc"], label="bc")
ax.semilogy(loss["iters"], loss["c"], label="continuity (mean r_c²)")
ax.set_xlabel("Adam iteration"); ax.set_ylabel("loss")
ax.set_title(f"Training loss — {args.tag} (final after L-BFGS: {float(loss['final_loss']):.3e})")
ax.grid(True, which="both", alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, f"{args.tag}_loss.png"), dpi=130)
plt.close()

print(f"saved plots: {args.tag}_fields.png, _vorticity_div.png, _centerlines.png, _loss.png")
