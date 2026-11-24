#!/bin/bash
# Train flocking with wind (native and rho) for every seed.
# Output: results/flocking/<native|robustness>/{agents,gifs,traces}/wind/flocking_<mode>_seed<N>.*
# Usage: bash flocking_wind.sh flocking_wind

RUN_ID="wind"
SEEDS="30 31 32 33 34"
N_AGENTS="3"

if [ -z "$1" ]; then
    echo "Error: No config file specified."
    echo "Usage: bash flocking_wind.sh <config_name_without_yaml>"
    return 1 2>/dev/null || exit 1
fi

CONFIG_PATH="${1%.yaml}.yaml"
mkdir -p "local_logs"

run_one() {
    local n_agents="$1" seed="$2"
    local name="${RUN_ID}_agents_${n_agents}_seed_${seed}"
    local log="local_logs/${name}.log"

    echo "[$(date +%H:%M:%S)] START $name -> $log"
    uv run -m src.experiments.experiment \
        --exp "$CONFIG_PATH" \
        --n_agents "$n_agents" \
        --seed "$seed" \
        --train "true" \
        --run-id "$RUN_ID" 2>&1 | tee "$log"
}

for n in $N_AGENTS; do
    for seed in $SEEDS; do
        run_one "$n" "$seed"
    done
done
