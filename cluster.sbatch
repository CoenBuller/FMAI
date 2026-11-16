#!/bin/bash
#SBATCH --job-name=fmai-compare
#SBATCH --partition=L40s_students
#SBATCH --nodelist=ceratanium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --output=fmai-%j.out
# Trains every (experiment x option x seed) as its own process, all in parallel on the one GPU, then builds one
# compare figure + table per experiment. Submit from your laptop with:  bash submit.sh
set -euo pipefail

# ------------------------------- what to run (edit here) -------------------------------
EXPERIMENTS="IPPO_individual.yaml experiment_1.yaml"   # files in src/experiments/
MODES="none rho:coen mixed:coen"                       # options: none | rho[:spec] | mixed[:spec]
SEEDS="1 2 3"
SMOOTH=20                                              # moving-average window (eval points) in the compare plots
EXTRA=""                                               # flags for every run, e.g. "--steps 2000000 --agents 4"
# ----------------------------------------------------------------------------------------

# Code, venv and uv cache on the node's local disk (/home quota is small); results on /data, so they are
# visible from the login node and survive a timeout. Per-run logs next to this script, readable from fs.cc.
WORK=/local/$USER/fmai-ass1
OUT=/data/$USER/fmai-results
LOGS=$SLURM_SUBMIT_DIR/logs/$SLURM_JOB_ID
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/local/$USER/uv-cache UV_PYTHON_INSTALL_DIR=/local/$USER/uv-python
export OMP_NUM_THREADS=1                               # one CPU thread per run, so the runs don't fight over cores
export PYTHONPATH=src                                  # code imports fmai, config, environment from src/
echo "Running on $(hostname); work dir $WORK; results in $OUT; logs in $LOGS"

# /local on ceratanium has been full before: stop now rather than fail halfway through the install
free_gb=$(df -BG --output=avail /local | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 20 ] || { echo "Only ${free_gb}G free on /local, aborting"; exit 1; }

mkdir -p "$WORK" "$OUT" "$LOGS"
rsync -a --delete --exclude .venv --exclude results --exclude logs "$SLURM_SUBMIT_DIR"/ "$WORK"/
cd "$WORK"
rm -rf results && ln -s "$OUT" results                 # RESULTS_DIR (results/...) is written straight to /data

# Install from uv.lock without qsfo (an unused local-path dependency that doesn't exist on the cluster)
uv export --frozen --no-hashes --no-emit-project --no-emit-package qsfo -o requirements.txt
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

nvidia-smi
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'"

# GIFs need a display: start a virtual one if Xvfb exists (save_gif only warns if rendering fails)
if command -v Xvfb >/dev/null; then
    export DISPLAY=:$(( SLURM_JOB_ID % 900 + 100 ))
    Xvfb "$DISPLAY" -screen 0 1024x768x24 >/dev/null 2>&1 &
fi

PY=".venv/bin/python -m experiments.experiment"
pids=()
for exp in $EXPERIMENTS; do
    rid="${SLURM_JOB_ID}_${exp%.yaml}"
    for mode in $MODES; do
        for seed in $SEEDS; do
            $PY --exp "$exp" --modes "$mode" --seed "$seed" --seeds 1 --run-id "$rid" $EXTRA \
                > "$LOGS/${rid}_${mode/:/_}_seed${seed}.log" 2>&1 &
            pids+=($!)
        done
    done
done
echo "Started ${#pids[@]} runs"

failed=0
for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
echo "$failed of ${#pids[@]} runs failed (see $LOGS)"

for exp in $EXPERIMENTS; do
    rid="${SLURM_JOB_ID}_${exp%.yaml}"
    $PY --exp "$exp" --run-id "$rid" --compare-only --smooth "$SMOOTH" $EXTRA || echo "compare failed for $exp"
done
echo "Results in $OUT/<RESULTS_DIR>/traces/${SLURM_JOB_ID}_*"
