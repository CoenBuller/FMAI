#!/bin/bash 

RESULTS_DIR="${RESULTS_DIR:-results/hyperparam}"
SEEDS="${SEEDS:-30 31 32 33 34 35}"
MAPPO="${MAPPO:-False}"
N_AGENTS="${N_AGENTS:-4 5 6 7 8}"

if [ -z "$1" ]; then
    echo "Error: No config file specified."
    echo "Usage: ./hyperparam.sh <config_name_without_yaml>"
    return 1 2>/dev/null || exit 1
fi

CONFIG_PATH="${1%.yaml}.yaml"
mkdir -p "local_logs"

run_one() {
    local mappo="$1" n_agents="$2" seed="$3" results="$4"
    local name="mappo_${mappo}_agents_${n_agents}_seed_${seed}"
    local log="local_logs/${name}.log"

    echo "[$(date +%H:%M:%S)] START $name -> $log"
    uv run -m src.experiments.experiment \
        --exp "$CONFIG_PATH" \
        --mappo "$mappo" \
        --n_agents "$n_agents" \
        --seed "$seed" \
        --results_dir "$results" 2>&1 | tee "$log"
}

for alg in $MAPPO; do
    for n in $N_AGENTS; do
        for seed in $SEEDS; do
            dir="$RESULTS_DIR/mappo_${alg}/agents_${n}/seed_${seed}"
            run_one "$alg" "$n" "$seed" "$dir"
        done
    done
done