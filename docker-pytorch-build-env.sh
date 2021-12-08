#!/bin/bash

# Build A Pytorch Docker Image
# - for kraken

TORCH_DOCKERFILE="https://raw.githubusercontent.com/pytorch/pytorch/4c4c03124bd246aa891dbb9c00c6bd8a03405796/docker/pytorch/ubuntu_cpu_gpu/Dockerfile"
IMAGE_NAME="ubuntu_1804_py_397_cuda_102_cudnn_8_dev"

nvidia-docker build "$TORCH_DOCKERFILE" \
    "$@" \
    -t "$IMAGE_NAME" \
    --build-arg BASE_IMAGE="nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04" \
    --build-arg PYTHON_VERSION="3.9.7" \
    --build-arg MAGMA_CUDA_VERSION="magma-cuda102" \
    --build-arg TORCH_CUDA_ARCH_LIST_VAR="3.7+PTX;5.0;6.0;6.1;7.0;7.5" \
