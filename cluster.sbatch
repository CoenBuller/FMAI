#!/bin/bash
#SBATCH --job-name=fmai-compare
#SBATCH --partition=L40s_students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --output=fmai-%j.out
# Trains every (experiment x option x seed) as its own process, all in parallel on the one GPU, then builds one
# compare figure + table and a robustness analysis (experiments/analyze.py) per experiment. Submit from your laptop with:  bash submit.sh
set -euo pipefail

# ------------------------------- what to run (edit here) -------------------------------
EXPERIMENTS="IPPO_zones.yaml flocking_wind.yaml"      # files in src/experiments/
MODES="none rho"                                       # none (native reward) | rho (robustness reward)
SEEDS="1 2 3 4 5"
SMOOTH=5                                               # moving-average window (eval points) in the compare plots
EXTRA=""                                               # flags for every run, e.g. "--seeds 2"
SCALE_AGENTS="2 4 5 6 7 8"                             # zero-shot scaling test on these numbers of agents ("" = skip)
ANALYSIS_EPISODES=20                                   # test envs per checkpoint for the robustness-over-time analysis
MAX_PARALLEL=4                                         # runs at once on the GPU (one CPU each; 30 at once ran out of its 44 GB)
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

# Install the exact versions of uv.lock
uv export --frozen --no-hashes --no-emit-project -o requirements.txt
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

nvidia-smi
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'"

# GIFs: there is no display on the node, so save_gif falls back to its matplotlib renderer.
# At most MAX_PARALLEL runs at a time; the next one starts as soon as one finishes.
PY=".venv/bin/python -m experiments.experiment"     # --train true: without it experiment.py runs the stress test
n=0; failed=0
for exp in $EXPERIMENTS; do
    rid="${SLURM_JOB_ID}_${exp%.yaml}"
    for mode in $MODES; do
        for seed in $SEEDS; do
            while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do wait -n || failed=$((failed + 1)); done
            gif=$([ "$seed" = "${SEEDS%% *}" ] && echo true || echo false)    # eval GIFs for the first seed only
            $PY --exp "$exp" --modes "$mode" --seed "$seed" --seeds 1 --eval_gif "$gif" --run-id "$rid" --train true $EXTRA \
                > "$LOGS/${rid}_${mode/:/_}_seed${seed}.log" 2>&1 &
            n=$((n + 1)); echo "Started run $n: $rid $mode seed $seed"
        done
    done
done
while [ -n "$(jobs -rp)" ]; do wait -n || failed=$((failed + 1)); done
echo "$failed of $n runs failed (see $LOGS)"

for exp in $EXPERIMENTS; do
    rid="${SLURM_JOB_ID}_${exp%.yaml}"
    RESULTS=$(grep -m1 RESULTS_DIR "src/experiments/$exp" | cut -d'"' -f2)    # results/navigation, results/flocking
    $PY --exp "$exp" --run-id "$rid" --compare-only --smooth "$SMOOTH" $EXTRA || echo "compare failed for $exp"
    .venv/bin/python -m experiments.analyze --results "$RESULTS" --run-id "$rid" --episodes "$ANALYSIS_EPISODES" || echo "analysis failed for $exp"
    [ -z "$SCALE_AGENTS" ] || .venv/bin/python -m experiments.scaling --results "$RESULTS" --run-id "$rid" \
        --n-agents $SCALE_AGENTS || echo "scaling test failed for $exp"
    # learning curves, final eval, steps to goal and spec rho per step of both options in one figure + table
    .venv/bin/python -m experiments.summary --smooth "$SMOOTH" --results "$RESULTS" --run-id "$rid" || echo "summary failed for $exp"
done
echo "Results in $OUT/{navigation,flocking}/: <native|robustness>/{agents,gifs,traces}/${SLURM_JOB_ID}_*, comparison/"
