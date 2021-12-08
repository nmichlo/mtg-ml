#!/bin/bash

docker build . -t mtgml39

docker run \
    --rm -it --gpus=all --shm-size=1G \
    -v ~/workspace/playground/mtg-ml/mtg_ml:/workspace/mtg_ml \
    -v ~/workspace/playground/mtg-ml/experiment:/workspace/experiment \
    -v ~/workspace/playground/mtg-ml/out:/workspace/out \
    -v ~/workspace/playground/mtg-dataset/data:/data \
    mtgml39 \
    "$@"
