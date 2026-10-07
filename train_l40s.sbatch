#!/bin/bash
#SBATCH --job-name=fmai-marl
#SBATCH --partition=L40s_students
#SBATCH --exclude=ceratanium           # its /local disk was full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --output=fmai-%j.out
set -e  # stop at the first failing setup command (e.g. a full disk) instead of continuing

# Runs to start, all in parallel on the one GPU (each algorithm/seed is its own process)
ALGORITHMS="mappo ippo maddpg matd3 masac"
SEEDS="0 1 2"

# Code, Python and packages live on this GPU machine's local disk because /home has a small quota;
# results go to /data.
WORK=/local/$USER/fmai
OUT=/data/$USER/fmai-results/$SLURM_JOB_ID
LOGS=$SLURM_SUBMIT_DIR/logs/$SLURM_JOB_ID   # one small progress log per run, readable from the login node
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/local/$USER/uv-cache UV_PYTHON_INSTALL_DIR=/local/$USER/uv-python
export OMP_NUM_THREADS=1  # one CPU thread per run, so 15 runs don't fight over cores
echo "Running on $(hostname); work dir $WORK; logs in $LOGS"

mkdir -p "$WORK" "$OUT" "$LOGS"
rsync -a --exclude .venv --exclude results --exclude logs "$SLURM_SUBMIT_DIR"/ "$WORK"/
cd "$WORK"
rm -rf results && ln -s "$OUT" results  # results are written straight to /data, so a timeout keeps them
uv add torchrl matplotlib
nvidia-smi

# Stop immediately if torch cannot see the GPU, instead of silently training on CPU for hours
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'"

for algo in $ALGORITHMS; do
  for seed in $SEEDS; do
    .venv/bin/python src/train.py "$algo" "$seed" > "$LOGS/${algo}_seed${seed}.log" 2>&1 &
  done
done
wait

.venv/bin/python src/plot.py
echo "Results in $OUT"
