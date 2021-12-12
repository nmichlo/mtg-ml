
# need to install python or conda
FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

# install cmake and build tools
RUN apt-get update
RUN apt-get install -y \
        build-essential \
        wget
        # cmake

# install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/root/miniconda3/bin:${PATH}"

# install the requirements
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY requirements-exp.txt .
RUN pip install -r requirements-exp.txt

RUN apt-get install -y cmake

# install horovod
RUN HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]


# ==================================== #

# doesnt work... we need cuda 10, this is cuda 11
#FROM pytorchlightning/pytorch_lightning:base-conda-py3.8-torch1.10
#RUN python -c "import torch; print(torch.cuda.is_available())"


#FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
#RUN conda  list
#RUN _pkgs="$(conda list | grep -v '^#' | grep -v '^_' | awk '{print $1}')" ; \
#    conda uninstall "$_pkgs" ;
#    conda install -y python=3.9.7 ; \
#    conda install -y "$_pkgs"


#FROM continuumio/miniconda3:4.10.3
#RUN apt-get update
#RUN apt-get install -y build-essential
#RUN conda install -y \
#    -c pytorch \
#    -c nvidia \
#        python=3.9.7 \
#        pytorch=1.9.1 \
#        torchvision=0.10.1 \
#        cudatoolkit=10.2.89 \
#        cuda-cupti=11.5.114 \
#        cuda-nvcc=11.5.119 \
#        numpy
#        # cudnn=8.2.1.32 \
#        # nccl \
#        # nvcc_linux-64 \
#        # mpi4py
#RUN HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]


## install the requirements
#COPY requirements.txt .
#RUN pip install -r requirements.txt
#COPY requirements-exp.txt .
#RUN pip install -r requirements-exp.txt
#
#RUN ls /usr/local/cuda
