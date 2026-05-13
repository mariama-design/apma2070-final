#!/bin/bash
#SBATCH --job-name=cavity_pinn_t1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=oscar/logs/task1_%j.out
#SBATCH --error=oscar/logs/task1_%j.err

# ---- environment ----
# Adjust to whatever your Oscar setup uses. Two common options:
#
# (a) Use the CCV-maintained TF module (simplest, no conda):
#       module load python/3.11 cuda/12.4 cudnn/9.0
#       python -m venv ~/envs/pinn
#       source ~/envs/pinn/bin/activate
#       pip install tensorflow scipy matplotlib
#
# (b) Or load a personal conda env:
#       module load anaconda/2023.09-0
#       source activate pinn
#
# Edit the next two lines to match your setup before submitting.
module load python/3.11.0s cuda/12.4 cudnn/9.0
source ~/envs/pinn/bin/activate

# Sanity: confirm GPU is visible.
python -c "import tensorflow as tf; print('TF', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

mkdir -p oscar/logs task1_forward/results

# ---- four Task 1 runs ----
# Run sequentially in one job so they share the loaded TF context.
echo "===== Run 1/4: baseline (vanilla tanh, uniform collocation) ====="
PYTHONUNBUFFERED=1 python -u task1_forward/train.py --tag baseline \
    --adam-iters 30000 --lbfgs-iters 5000

echo "===== Run 2/4: +LAAF only ====="
PYTHONUNBUFFERED=1 python -u task1_forward/train.py --tag laaf --laaf \
    --adam-iters 30000 --lbfgs-iters 5000

echo "===== Run 3/4: +RAR only ====="
PYTHONUNBUFFERED=1 python -u task1_forward/train.py --tag rar --rar \
    --adam-iters 30000 --lbfgs-iters 5000

echo "===== Run 4/4: +both (LAAF + RAR) ====="
PYTHONUNBUFFERED=1 python -u task1_forward/train.py --tag both --laaf --rar \
    --adam-iters 30000 --lbfgs-iters 5000

echo "===== Step 6: failure-mode (baseline with continuity weight cut 100x) ====="
PYTHONUNBUFFERED=1 python -u task1_forward/train.py --tag failure --lambda-c 0.01 \
    --adam-iters 30000 --lbfgs-iters 5000

echo "===== all Task 1 runs done ====="
