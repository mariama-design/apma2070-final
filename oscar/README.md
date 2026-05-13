# Running on Oscar

CPU on my MacBook is too slow for the four-run Task 1 sweep (~5 hr). On an Oscar GPU
node the same job should be **~10–30 min total**.

## One-time setup on Oscar

```bash
# ssh in
ssh <username>@ssh.ccv.brown.edu

# pick a place to live (home or scratch — scratch is faster but periodic-wiped)
cd ~        # or: cd /scratch/$USER

# get a venv with TensorFlow + scipy + matplotlib
module load python/3.11.0s cuda/12.4 cudnn/9.0
python -m venv ~/envs/pinn
source ~/envs/pinn/bin/activate
pip install --upgrade pip
pip install tensorflow scipy matplotlib
```

If your Oscar account has a different cuda/cudnn version, swap the module names
in `run_task1.sbatch` accordingly. `module avail cuda` lists what is available.

Confirm TF sees the GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# should print: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Copy the repo to Oscar

From this laptop:

```bash
cd "/Users/mayan/Downloads/apma 2070 final"
rsync -avL --exclude='cavity_flow.gif' --exclude='__pycache__' \
    ./ <username>@ssh.ccv.brown.edu:~/apma2070/
```

`-L` makes rsync follow symlinks, so the actual `.mat` files in `data/` get copied
instead of dangling links.

## Submit the job

On Oscar:

```bash
cd ~/apma2070
sbatch oscar/run_task1.sbatch
squeue -u $USER          # check it's queued or running
tail -f oscar/logs/task1_*.out   # follow training output live
```

The script runs all four Task 1 ablations (baseline, +LAAF, +RAR, +both) plus
the failure-mode case sequentially in one job. Results land in
`task1_forward/results/`.

## Pull results back

When the job finishes:

```bash
# from this laptop
cd "/Users/mayan/Downloads/apma 2070 final"
rsync -av <username>@ssh.ccv.brown.edu:~/apma2070/task1_forward/results/ \
    task1_forward/results/
```

Then run the plot script locally (no GPU needed for plotting):

```bash
python3 task1_forward/plots.py --tag baseline
python3 task1_forward/plots.py --tag laaf
python3 task1_forward/plots.py --tag rar
python3 task1_forward/plots.py --tag both
python3 task1_forward/plots.py --tag failure
```

## Quick interactive test before committing 4 hr

If you want to sanity-check the GPU pipeline first, grab a 30-min interactive
GPU and run a short version:

```bash
interact -n 4 -t 0:30:00 -p gpu --gres=gpu:1
module load python/3.11.0s cuda/12.4 cudnn/9.0
source ~/envs/pinn/bin/activate
cd ~/apma2070
python -u task1_forward/train.py --tag _gpu_test --adam-iters 2000 --lbfgs-iters 0
```

If that runs in a couple of minutes, the full job will be fine.

## SLURM gotchas

- `#SBATCH --time=04:00:00` is a hard limit; pad it. If the job is killed, you
  can resume from the last saved weights by adding a `--resume` flag (not
  implemented yet — current scripts always start fresh).
- If you don't have GPU allocation, drop `--partition=gpu --gres=gpu:1` and
  it will run on CPU (slow, same as my laptop).
- `oscar/logs/` is created by the SLURM script — make sure it exists or
  the `--output=` redirect will fail silently.
