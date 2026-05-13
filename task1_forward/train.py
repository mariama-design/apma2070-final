"""Task 1 forward PINN for steady lid-driven cavity at Re=100.

Loss = PDE residuals (mom-x, mom-y, continuity) + boundary conditions + pressure gauge.
No data loss in this task.

Same script handles all four configurations via flags:
  --tag baseline                 (vanilla tanh, uniform collocation)
  --tag laaf  --laaf             (locally adaptive activation)
  --tag rar   --rar              (residual-based adaptive refinement)
  --tag both  --laaf --rar       (both)

Also supports --lambda-c for the failure-mode case (cut continuity weight by ~100x).

Outputs:
  results/<tag>.weights.h5   network weights
  results/<tag>_field.npz    u, v, p on a 201x201 grid
  results/<tag>_loss.npz     training loss history
"""

import argparse
import os
import sys
import time
import numpy as np
import scipy.optimize
import tensorflow as tf

# Make utils.py importable from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, required=True)
parser.add_argument("--laaf", action="store_true")
parser.add_argument("--rar", action="store_true")
parser.add_argument("--adam-iters", type=int, default=30000)
parser.add_argument("--lbfgs-iters", type=int, default=5000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lambda-c", type=float, default=1.0,
                    help="continuity loss weight (set <<1 for the failure-mode run)")
args = parser.parse_args()

np.random.seed(args.seed)
tf.random.set_seed(args.seed)

RE = 100.0
N_INT = 10000          # interior collocation points
N_WALL = 100           # per-wall BC points
GAUGE_N = 256          # points for the zero-mean pressure gauge
LAMBDA_BC = 10.0       # BC weight (rest are 1.0)
LR_SCHEDULE = {0: 1e-3, 10000: 5e-4, 20000: 1e-4}
RESAMPLE_INTERVAL = 1000
RAR_INTERVAL = 2000
K_RAR = 1000
N_RAR_POOL = 50000


class PINN(tf.keras.Model):
    """MLP with tanh activations, optionally locally adaptive (LAAF)."""
    def __init__(self, hidden=[20] * 9, laaf=False):
        super().__init__()
        self.hidden_layers_ = [
            tf.keras.layers.Dense(n, kernel_initializer="glorot_normal") for n in hidden
        ]
        self.out_layer = tf.keras.layers.Dense(3, kernel_initializer="glorot_normal")
        self.laaf = laaf
        if laaf:
            self.a_params = [
                tf.Variable(1.0, trainable=True, dtype=tf.float32, name=f"laaf_a{i}")
                for i in range(len(hidden))
            ]

    def call(self, xy):
        h = xy
        for i, layer in enumerate(self.hidden_layers_):
            h = layer(h)
            if self.laaf:
                h = tf.tanh(self.a_params[i] * h)
            else:
                h = tf.tanh(h)
        return self.out_layer(h)


model = PINN(laaf=args.laaf)
_ = model(tf.zeros((1, 2)))  # build


def predict_uvp(x, y):
    out = model(tf.concat([x, y], axis=1))
    return out[:, 0:1], out[:, 1:2], out[:, 2:3]


def pde_residuals(x, y):
    with tf.GradientTape(persistent=True) as t2:
        t2.watch([x, y])
        with tf.GradientTape(persistent=True) as t1:
            t1.watch([x, y])
            u, v, p = predict_uvp(x, y)
        u_x = t1.gradient(u, x); u_y = t1.gradient(u, y)
        v_x = t1.gradient(v, x); v_y = t1.gradient(v, y)
        p_x = t1.gradient(p, x); p_y = t1.gradient(p, y)
        del t1
    u_xx = t2.gradient(u_x, x); u_yy = t2.gradient(u_y, y)
    v_xx = t2.gradient(v_x, x); v_yy = t2.gradient(v_y, y)
    del t2
    ru = u * u_x + v * u_y + p_x - (1.0 / RE) * (u_xx + u_yy)
    rv = u * v_x + v * v_y + p_y - (1.0 / RE) * (v_xx + v_yy)
    rc = u_x + v_y
    return ru, rv, rc


def sample_interior(n):
    xy = np.random.rand(n, 2).astype(np.float32)
    return xy[:, 0:1], xy[:, 1:2]


# --- boundary points (fixed once) ---
t_wall = np.linspace(0.0, 1.0, N_WALL, dtype=np.float32).reshape(-1, 1)
zeros = np.zeros_like(t_wall)
ones = np.ones_like(t_wall)
bc_pts = {
    "bot":   (t_wall, zeros),
    "top":   (t_wall, ones),
    "left":  (zeros, t_wall),
    "right": (ones,  t_wall),
}
corner_x = np.array([[0.0], [1.0]], dtype=np.float32)  # lid corners (0,1) and (1,1)
corner_y = np.array([[1.0], [1.0]], dtype=np.float32)
gauge_x = np.random.rand(GAUGE_N, 1).astype(np.float32)
gauge_y = np.random.rand(GAUGE_N, 1).astype(np.float32)

# Convert to tf constants.
bc_tf = {k: (tf.constant(x), tf.constant(y)) for k, (x, y) in bc_pts.items()}
corner_x_tf = tf.constant(corner_x); corner_y_tf = tf.constant(corner_y)
gauge_x_tf = tf.constant(gauge_x); gauge_y_tf = tf.constant(gauge_y)

# Interior collocation set (a Variable so we can reassign during training).
xi_np, yi_np = sample_interior(N_INT)
xi_var = tf.Variable(xi_np); yi_var = tf.Variable(yi_np)


def compute_loss(x_int, y_int):
    ru, rv, rc = pde_residuals(x_int, y_int)
    loss_pde_u = tf.reduce_mean(ru ** 2)
    loss_pde_v = tf.reduce_mean(rv ** 2)
    loss_pde_c = tf.reduce_mean(rc ** 2)
    loss_pde = loss_pde_u + loss_pde_v + args.lambda_c * loss_pde_c

    # BCs: no-slip on bottom/left/right; lid: u=1, v=0; lid corners pinned to u=0.
    loss_bc = tf.zeros((), dtype=tf.float32)
    for k, (bx, by) in bc_tf.items():
        u, v, _ = predict_uvp(bx, by)
        if k == "top":
            loss_bc += tf.reduce_mean((u - 1.0) ** 2) + tf.reduce_mean(v ** 2)
        else:
            loss_bc += tf.reduce_mean(u ** 2) + tf.reduce_mean(v ** 2)
    u_c, v_c, _ = predict_uvp(corner_x_tf, corner_y_tf)
    loss_bc += tf.reduce_mean(u_c ** 2) + tf.reduce_mean(v_c ** 2)

    # Pressure gauge: zero-mean over a fixed sample.
    _, _, p_g = predict_uvp(gauge_x_tf, gauge_y_tf)
    loss_p = tf.reduce_mean(p_g) ** 2

    total = loss_pde + LAMBDA_BC * loss_bc + loss_p
    return total, {
        "pde": float(loss_pde),
        "bc":  float(loss_bc),
        "p":   float(loss_p),
        "c":   float(loss_pde_c),
    }


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


@tf.function
def adam_step(x_int, y_int):
    with tf.GradientTape() as g:
        ru, rv, rc = pde_residuals(x_int, y_int)
        loss_pde_u = tf.reduce_mean(ru ** 2)
        loss_pde_v = tf.reduce_mean(rv ** 2)
        loss_pde_c = tf.reduce_mean(rc ** 2)
        loss_pde = loss_pde_u + loss_pde_v + args.lambda_c * loss_pde_c

        loss_bc = tf.zeros((), dtype=tf.float32)
        for k, (bx, by) in bc_tf.items():
            u, v, _ = predict_uvp(bx, by)
            if k == "top":
                loss_bc += tf.reduce_mean((u - 1.0) ** 2) + tf.reduce_mean(v ** 2)
            else:
                loss_bc += tf.reduce_mean(u ** 2) + tf.reduce_mean(v ** 2)
        u_c, v_c, _ = predict_uvp(corner_x_tf, corner_y_tf)
        loss_bc += tf.reduce_mean(u_c ** 2) + tf.reduce_mean(v_c ** 2)

        _, _, p_g = predict_uvp(gauge_x_tf, gauge_y_tf)
        loss_p = tf.reduce_mean(p_g) ** 2

        total = loss_pde + LAMBDA_BC * loss_bc + loss_p
    grads = g.gradient(total, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total


# ---------- Adam phase ----------
print(f"Adam phase: {args.adam_iters} iters | laaf={args.laaf} rar={args.rar} lambda_c={args.lambda_c}")
history = []
t0 = time.time()
for it in range(args.adam_iters + 1):
    if it in LR_SCHEDULE:
        optimizer.learning_rate.assign(LR_SCHEDULE[it])

    if it > 0 and it % RESAMPLE_INTERVAL == 0:
        if args.rar and it % RAR_INTERVAL == 0:
            xp_np, yp_np = sample_interior(N_RAR_POOL)
            xp = tf.constant(xp_np); yp = tf.constant(yp_np)
            ru, rv, rc = pde_residuals(xp, yp)
            r_mag = (ru ** 2 + rv ** 2 + rc ** 2).numpy().ravel()
            top_idx = np.argpartition(-r_mag, K_RAR)[:K_RAR]
            xu_np, yu_np = sample_interior(N_INT - K_RAR)
            new_x = np.vstack([xu_np, xp_np[top_idx]]).astype(np.float32)
            new_y = np.vstack([yu_np, yp_np[top_idx]]).astype(np.float32)
        else:
            new_x, new_y = sample_interior(N_INT)
        xi_var.assign(new_x); yi_var.assign(new_y)

    loss_val = adam_step(xi_var, yi_var)

    if it % 500 == 0:
        _, parts = compute_loss(xi_var, yi_var)
        print(f"  it {it:6d}  loss={float(loss_val):.3e}  pde={parts['pde']:.3e}  "
              f"bc={parts['bc']:.3e}  c={parts['c']:.3e}  p={parts['p']:.3e}  "
              f"t={time.time() - t0:.1f}s")
        history.append((it, float(loss_val), parts))

print(f"Adam done. Elapsed {time.time() - t0:.1f}s")


# ---------- L-BFGS phase (scipy) ----------
def get_flat():
    return tf.concat([tf.reshape(v, [-1]) for v in model.trainable_variables], axis=0).numpy().astype(np.float64)

def set_flat(x):
    idx = 0
    for v in model.trainable_variables:
        sz = int(np.prod(v.shape))
        v.assign(tf.reshape(tf.cast(x[idx:idx + sz], dtype=v.dtype), v.shape))
        idx += sz

def loss_and_grad(x):
    set_flat(x)
    with tf.GradientTape() as g:
        total, _ = compute_loss(xi_var, yi_var)
    grads = g.gradient(total, model.trainable_variables)
    flat = tf.concat([tf.reshape(gr, [-1]) for gr in grads], axis=0).numpy().astype(np.float64)
    return float(total), flat

print(f"L-BFGS phase: maxiter={args.lbfgs_iters}")
t0 = time.time()
cb_counter = [0]
def cb(xk):
    cb_counter[0] += 1
    if cb_counter[0] % 100 == 0:
        l, _ = loss_and_grad(xk)
        print(f"  lbfgs iter {cb_counter[0]:5d}  loss={l:.3e}")

x0 = get_flat()
res = scipy.optimize.minimize(
    loss_and_grad, x0, jac=True, method="L-BFGS-B",
    options={"maxiter": args.lbfgs_iters, "ftol": 1e-12, "gtol": 1e-10, "maxcor": 50, "disp": False},
    callback=cb,
)
set_flat(res.x)
print(f"L-BFGS done. final loss={res.fun:.3e}  niter={res.nit}  t={time.time() - t0:.1f}s")


# ---------- Save ----------
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

model.save_weights(os.path.join(RESULTS, f"{args.tag}.weights.h5"))

n = 201
xg = np.linspace(0.0, 1.0, n).astype(np.float32)
yg = np.linspace(0.0, 1.0, n).astype(np.float32)
Xg, Yg = np.meshgrid(xg, yg)
xv = Xg.ravel()[:, None]; yv = Yg.ravel()[:, None]
u_p, v_p, p_p = predict_uvp(tf.constant(xv), tf.constant(yv))
U_pred = u_p.numpy().reshape(n, n)
V_pred = v_p.numpy().reshape(n, n)
P_pred = p_p.numpy().reshape(n, n)
np.savez(os.path.join(RESULTS, f"{args.tag}_field.npz"),
         xi=xg, yi=yg, U=U_pred, V=V_pred, P=P_pred)

np.savez(os.path.join(RESULTS, f"{args.tag}_loss.npz"),
         iters=np.array([h[0] for h in history]),
         loss=np.array([h[1] for h in history]),
         pde=np.array([h[2]["pde"] for h in history]),
         bc=np.array([h[2]["bc"] for h in history]),
         c=np.array([h[2]["c"] for h in history]),
         p=np.array([h[2]["p"] for h in history]),
         final_loss=float(res.fun))

print(f"saved to {RESULTS}/{args.tag}.weights.h5, {args.tag}_field.npz, {args.tag}_loss.npz")
