#!/bin/bash 

# Move up one directory level to the parent folder
cd ..

# Clone second repository which is needed for online monitoring. 
git clone https://github.com/CoenBuller/artifact-fm-qsfo.git

cd FMAI

# Install dependancies and activate environment
uv sync
source .venv/bin/activate