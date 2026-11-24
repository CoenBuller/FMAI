#!/bin/bash

RESULTS_DIR="results_2/flocking"
SEEDS="30 31 32 33 34"
N_AGENTS="3"

if [ -z "$1" ]; then
    echo "Error: No config file specified."
    echo "Usage: ./hyperparam.sh <config_name_without_yaml>"
    return 1 2>/dev/null || exit 1
fi

CONFIG_PATH="${1%.yaml}.yaml"
mkdir -p "local_logs"

run_one() {
    local n_agents="$1" seed="$2" results="$3"
    local name="agents_${n_agents}_seed_${seed}"
    local log="local_logs/${name}.log"

    echo "[$(date +%H:%M:%S)] START $name -> $log"
    uv run -m src.experiments.experiment \
        --exp "$CONFIG_PATH" \
        --n_agents "$n_agents" \
        --seed "$seed" \
        --train "true" \
        --results_dir "$results" 2>&1 | tee "$log"
}

for n in $N_AGENTS; do
    for seed in $SEEDS; do
        dir="$RESULTS_DIR/seed_${seed}"
        run_one "$n" "$seed" "$dir"
    done
done