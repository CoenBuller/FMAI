#!/bin/bash 

if [ -z "$1" ]; then
    echo "Error: No config file specified."
    echo "Usage: source run.sh <path_to_yaml>"
    return 1 2>/dev/null || exit 1
fi

CONFIG_PATH="$1.yaml"

# Forward the path to your python module
uv run -m src.experiments.experiment --exp "$CONFIG_PATH"
