#!/bin/bash

# build the docker image
docker build . -t mtgml39

# Run the docker command
docker run \
    --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --rm -it \
    -v ~/workspace/playground/mtg-ml/mtg_ml:/workspace/mtg_ml \
    -v ~/workspace/playground/mtg-ml/experiment:/workspace/experiment \
    -v ~/workspace/playground/mtg-ml/out:/workspace/out \
    -v ~/workspace/playground/mtg-dataset/data:/data \
    mtgml39 \
    "$@"
