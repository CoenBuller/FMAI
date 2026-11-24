#!/bin/bash

RESULTS_DIR="results_2"
SEEDS="30 31 32 33 34"
MODES="rho none"
N_AGENTS="4 5 6"
WIND_TRAINED="True False"
WIND_STRENGTH="0.17 0.19 0.22 0.25"
WIND_GUSTS="0.17 0.19 0.22 0.25"

if [ -z "$1" ]; then
    echo "Error: No config file specified."
    echo "Usage: ./stresstest.sh <config_name_without_yaml>"
    return 1 2>/dev/null || exit 1
fi

CONFIG_PATH="${1%.yaml}.yaml"
mkdir -p "local_logs"

run_one() {
    local mode="$1"
    local n_agents="$2"
    local seed="$3"
    local results="$4"
    local wind_trained="$5"
    local wind_strength="$6"
    local wind_gusts="$7"

    local name="${mode}_trained_${wind_trained}_n_${n_agents}_w_${wind_strength}_g_${wind_gusts}_seed_${seed}"
    local log="local_logs/${name}.log"

    echo "[$(date +%H:%M:%S)] START $name -> $log"
    uv run -m src.experiments.experiment \
        --exp "$CONFIG_PATH" \
        --train False \
        --mode "$mode" \
        --n_agents "$n_agents" \
        --seed "$seed" \
        --wind_trained "$wind_trained" \
        --wind_strength "$wind_strength" \
        --wind_gust_strength "$wind_gusts" \
        --results_dir "$results" 2>&1 | tee "$log"
}

for mode in $MODES; do
    for n in $N_AGENTS; do
        for seed in $SEEDS; do
            for trained in $WIND_TRAINED; do
                for wind_strength in $WIND_STRENGTH; do
                    for wind_gust in $WIND_GUSTS; do
                        run_one "$mode" "$n" "$seed" "$RESULTS_DIR" "$trained" "$wind_strength" "$wind_gust"
                    done
                done
            done
        done
    done
done
