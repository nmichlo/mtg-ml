#!/bin/bash

# get number of GPUS
_num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Running With At Least 1 of $_num_gpus GPUs:"
nvidia-smi --query-gpu=name --format=csv,noheader

# Run the docker command
docker run \
    --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --rm -it \
    -v ~/workspace/playground/mtg-ml/hydra_plugins:/workspace/hydra_plugins \
    -v ~/workspace/playground/mtg-ml/mtg_ml:/workspace/mtg_ml \
    -v ~/workspace/playground/mtg-ml/experiment:/workspace/experiment \
    -v ~/workspace/playground/mtg-ml/out:/workspace/out \
    -v ~/workspace/playground/mtg-dataset/data:/data \
    --env PYTHONPATH=/workspace \
    mtgml39 \
    "$@"
