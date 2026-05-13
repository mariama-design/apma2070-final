# Plan

## Stack

- **Python + TensorFlow + NumPy + SciPy + Matplotlib.** Picked once. No DeepXDE / Modulus / PyTorch.
- One Python script per task, run from the repo root. Seeds fixed (`tf.random.set_seed`, `np.random.seed`).
- All trained weights and sampled point indices saved as `.h5` / `.npz` so plots can be redrawn without retraining.

## Repo layout

```
apma 2070 final/
├── data/                       # symlinks to the two provided datasets
│   ├── velocity.mat            # -> Re_100/velocity.mat
│   ├── pressure.mat            # -> Re_100/pressure.mat
│   └── cavity_data.mat         # -> Course project_.../Cavity_data/cavity_data.mat
├── task1_forward/
│   ├── train.py                # forward PINN, save weights
│   ├── plots.py                # contours, vorticity, centerlines, errors
│   ├── failure_mode.py         # the "looks right, is wrong" example
│   └── results/                # .h5 weights + .npz arrays + .png figures
├── task2_reconstruction/
│   ├── train.py                # loops N = 10, 20, 30, 40
│   ├── plots.py                # error heatmaps + table
│   └── results/
├── task3_sampling_patch/
│   ├── train.py                # uniform vs LHS, patch A vs patch B
│   ├── plots.py
│   └── results/
├── task4_noise/
│   ├── train.py                # noise 1% / 5%, baseline + Huber variant
│   ├── plots.py
│   └── results/
├── task5_re_sensitivity/       # mostly written, optional small experiment
├── report/
│   ├── report.tex              # 8–10 page write-up
│   └── figures/                # symlinks to selected results/*.png
└── utils.py                    # small: data loading + L2 error + vorticity + divergence
```

`utils.py` stays small on purpose (loader + a couple of metric functions). Each task has its own `train.py` and its own copy of the PINN class — duplicating ~30 lines is fine and keeps each task readable in isolation.

## Baseline PINN setup (used first in every task)

- **Network**: MLP, inputs `(x, y)`, outputs `(u, v, p)`, 9 hidden layers × 20 neurons, `tanh` activation, Xavier init.
- **Collocation points**: ~10 000 interior points sampled **uniformly at random**, refreshed every few thousand steps; ~400 points per wall for BC loss.
- **PDE residuals** (steady NS at Re=100):
  - $r_u = u u_x + v u_y + p_x - (1/Re)(u_{xx} + u_{yy})$
  - $r_v = u v_x + v v_y + p_y - (1/Re)(v_{xx} + v_{yy})$
  - $r_c = u_x + v_y$
- **Boundary loss**: lid ($y=1$): $u=1, v=0$; other three walls: $u=v=0$. Lid corners $u(0,1)=u(1,1)=0$ added explicitly so the singularity is regularized, not pretended away.
- **Pressure gauge**: penalize $(\text{mean}_\Omega p)^2$ — robust to a single-point pin and matches how the reference data normalizes.
- **Loss**: $\mathcal{L} = \lambda_{pde}\,\|r\|^2 + \lambda_{bc}\,\|bc\|^2 + \lambda_{p}\,(\bar p)^2 \;(+\; \lambda_{data}\,\|u-u_{obs}\|^2$ from Task 2 onward$)$.
- **Optimizer**: Adam ~30k iters with stepwise decay, then ~5k L-BFGS finetune. 

This is the **baseline configuration** — what every task is first solved with, so the report can show what is gained by the two additions below.

## Two enhancements (added after the baseline works, ablated separately)

Once the baseline forward solve in Task 1 is producing reasonable contours and centerline profiles, two changes are added — **independently first, then together** — so the report can attribute any gain to the right cause.

1. **Locally adaptive activation function (LAAF, Jagtap & Karniadakis 2020)**. Replace `tanh(x)` with `tanh(a · x)` where `a` is a trainable scalar per hidden layer (initialized to 1). ~5 lines of code. Expected effect: faster Adam convergence, slightly lower final loss.
2. **Residual-based Adaptive Refinement (RAR)**. Every ~2000 Adam steps: draw a large candidate pool (~50 000 random interior points), evaluate the PDE residual on them, keep the top-$k$ (e.g., $k = 1000$) and append them to the collocation set. Expected effect: concentrates points in the lid shear layer; lowers the local residual there.

This gives a four-cell ablation in Task 1: **baseline**, **+LAAF only**, **+RAR only**, **+both**. From Task 2 onward, the "+both" configuration is used as the default unless an ablation is specifically called out.


## Diagnostics (set in Task 0, reused everywhere)

1. Relative $L^2$ error on $u, v, p$ versus the spectral reference, computed on the full 201×201 grid.
2. Divergence norm $\|\nabla\cdot\mathbf{u}\|_{L^2(\Omega)}$, evaluated by autograd on the same grid.
3. Vorticity field $\omega = v_x - u_y$ — its sign pattern and extrema location flag missing corner eddies.
4. Centerline profiles $u(x, 0.5)$ and $v(0.5, y)$ overlaid on Ghia-style reference (extracted from the provided spectral data).

## Execution order

The plan is organized in the order things actually get run. Each step lists **goal → run → save → plot/report**.

### Step 0 — Setup
- **Goal**: data loaded and verified against the project figure.
- **Run**: write `utils.py` (load `.mat`, reshape to 201×201, relative $L^2$, vorticity, divergence). Quick script to plot $u, v, p$ from the spectral data.
- **Save**: nothing trained yet.
- **Plot**: one figure reproducing the project's Figure 2 to confirm the dataset is correct.

### Step 1 — Task 0 physics audit (written, no code)
- **Goal**: lock in the conceptual story and the diagnostics list before touching any PINN code.
- Cover: primary recirculation cell + weak bottom-corner eddies + lid shear layer; which features are BC-driven (lid motion sets rotation; no-slip on three walls forces corner stagnation) vs advection-driven (vortex-center drift to the right of geometric center, sharpened shear layer); pressure gauge argument ($\nabla p$ invariant under $p \to p + c$, impose $\int_\Omega p = 0$); the four diagnostics from the section above.

### Step 2 — Task 1 baseline forward PINN
- **Goal**: clean Re=100 forward solve with no data loss. The engineering core of the project.
- **Run**: `task1_forward/train.py --tag baseline` — vanilla `tanh`, uniform collocation, Adam → L-BFGS.
- **Save**: `results/baseline.h5` (weights), `results/baseline_field.npz` (predictions on 201×201 grid), `results/baseline_loss.npz` (training curve).
- **Plot**: $u, v, p$ contours, vorticity, centerline overlays vs spectral. Annotate relative $L^2$ for $u, v, p$ and the divergence norm.

### Step 3 — Task 1 +LAAF only
- **Goal**: isolate the effect of adaptive activation.
- **Run**: `train.py --tag laaf --laaf` (same seed as baseline).
- **Save**: `results/laaf.h5`, `_field.npz`, `_loss.npz`.
- Feeds into the Step 5 ablation table.

### Step 4 — Task 1 +RAR only
- **Goal**: isolate the effect of adaptive collocation.
- **Run**: `train.py --tag rar --rar` (same seed as baseline).
- **Save**: same pattern, plus the final collocation point set so the lid-layer concentration can be visualized.
- **Plot**: scatter of the final collocation points — expected to cluster under the lid.

### Step 5 — Task 1 +both (locked in as default for Tasks 2–4)
- **Goal**: final enhanced configuration.
- **Run**: `train.py --tag both --laaf --rar`.
- **Save**: same pattern.
- **Plot**: the four-config ablation panel:
  - Table of relative $L^2$ on $u, v, p$ + divergence norm for {baseline, +LAAF, +RAR, +both}.
  - Loss curves overlaid on one figure (four lines).
  - Side-by-side error heatmap row for baseline vs +both.
- Short paragraph attributing the gains to LAAF (smoother convergence) and RAR (lower residual near the lid) independently, and noting whether the combination is additive or interacts.

### Step 6 — Task 1 failure-mode
- **Goal**: the required "visually plausible but wrong" example.
- **Run**: `failure_mode.py` — same baseline script, `λ_continuity` cut by ~100×.
- **Save**: weights + field + divergence map.
- **Plot**: $u, v$ contours that look fine next to a divergence heatmap that does not. Diagnostic that catches it: divergence norm + divergence heatmap.

### Step 7 — Task 2 sparse reconstruction
- **Goal**: reconstruction error and divergence vs sample count $N$ on the +both setup.
- **Run**: `task2_reconstruction/train.py`. Fix patch $\Omega_p = [0.3, 0.7]^2$. For $N \in \{10, 20, 30, 40\}$, sample $(x_i, y_i) \in \Omega_p$ uniformly random (fixed seed); read $u, v, p$ at the nearest spectral grid point; add a data loss to the PINN loss.
- **Save**: one weights file per $N$ + the sampled point indices.
- **Plot**: per-$N$ table of relative $L^2$ on $u, v, p$ + $\|\nabla\cdot\mathbf{u}\|_{L^2}$; error heatmaps for $u, v, p$.
- **Discuss**: expect errors largest in the lid shear layer (steep gradients, far from any sample if patch is central) and mildly at the corners.

### Step 8 — Task 3 sampling × patch
- **Goal**: which patch and which sampler give the most information at fixed $N$.
- **Run**: `task3_sampling_patch/train.py` with $N = 30$. Two patches: **A** centered $[0.4, 0.6]^2$, **B** lid shear $[0.3, 0.7] \times [0.82, 0.95]$. Two samplers: uniform random, Latin Hypercube (`scipy.stats.qmc.LatinHypercube`). 2×2 grid of runs.
- **Save**: weights + sample indices per cell of the grid.
- **Plot**: 2×2 results panel (rel-$L^2$ and divergence in each cell); error heatmaps for best vs worst cell.
- **Discuss**: tie back to Task 0 diagnostics — patch B should constrain the global flow more (largest viscous + advective scales); LHS gives slightly lower variance across seeds, especially for small $N$.

### Step 9 — Task 4 noise + robustness
- **Goal**: degradation under noisy data, then one robustness modification.
- **Run**: `task4_noise/train.py`. Take Task 2's best $N$ (likely 30 or 40, central patch). Two noise levels: $\sigma = 0.01\,\|s\|_\infty$ and $\sigma = 0.05\,\|s\|_\infty$ per channel, additive Gaussian. Two loss variants on the data term: MSE (baseline), Huber ($\delta = $ a few × $\sigma$).
- **Save**: weights for each (noise level × loss) cell.
- **Plot**: error and divergence vs noise level for both losses; side-by-side $u$ heatmaps at 5% noise (MSE vs Huber).

### Step 10 — Task 5 Re sensitivity (mostly written)
- **Goal**: conceptual answer + optional small demo.
- **Run**: writing first. Re=100 → 1000: BL thickens with $1/\sqrt{Re}$ scaling under the lid, primary vortex center drifts toward the geometric center, secondary corner eddies grow and a small upstream-corner eddy appears. Strategy to infer unknown Re from sparse data: treat $Re$ (or $1/Re$) as a trainable scalar in the PDE loss, optimize jointly with the network weights; most informative sensor placement is where viscous terms dominate (right beneath the lid and in the bottom corners). This is the inverse-problem mode from Raissi et al.
- **Optional**: re-purpose the Task 2 script to leave $1/Re$ trainable and check it converges to a known synthetic value.

### Step 11 — Report
- Pull figures from each `results/` folder, write captions and analysis. Page budget below.

## Report (8–10 pages)

1. **Intro + governing equations** (½ page)
2. **Task 0 — physics audit & diagnostics** (1 page)
3. **Task 1 — forward PINN** (2 pages: baseline contours and centerlines vs spectral; 4-config ablation table + loss-curve plot for {baseline, +LAAF, +RAR, +both}; failure-mode panel)
4. **Task 2 — sparse reconstruction** (2 pages: $N$-sweep table + error-heatmap grid)
5. **Task 3 — sampling & patch** (1½ pages: 2×2 results panel + discussion of which region informs the global flow more)
6. **Task 4 — noise robustness** (1 page: error vs noise table, Huber vs MSE)
7. **Task 5 — Re sensitivity** (½–1 page: argument + optional figure)
8. **Conclusion / limits of reconstruction** (½ page)

Captions interpret the figure (what it shows + why it looks that way), not restate axes.

## Risks / things to watch

- Pure PDE+BC PINN at Re=100 is known to be slow to converge if the lid corners aren't handled — keep the explicit corner BC samples and consider a smooth lid profile (e.g., $u_{lid}(x) = 1 - (2x - 1)^{50}$) as a fallback only if vanilla doesn't converge.
- Loss weights matter. If results are off, rescale $\lambda_{bc}$ and $\lambda_{pde}$ before adding complexity; don't reach for fancy adaptive-weight schemes until needed.
- For Task 2 with $N = 10$, the run is genuinely under-constrained near the lid — that's the point of the exercise, not a bug.
- Pressure has the largest relative error in every PINN paper on this problem; report it as a finding, not a failure.
