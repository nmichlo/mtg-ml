
# need to install python or conda
FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

# install cmake and build tools
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        wget \
        cmake

# install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/root/miniconda3/bin:${PATH}"

# install torch
#RUN pip3 install torch==1.9.1+cu102 torchvision==0.10.1+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html

#RUN conda install -y \
#    -c pytorch \
#        pytorch=1.9.1 \
#        torchvision=0.10.1 \
#        cudatoolkit=10.2

# install the requirements
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY requirements-exp.txt .
RUN pip install -r requirements-exp.txt

# install horovod
RUN HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]

WORKDIR /workspace

# ALMOST WORKING BUT LIB MISMATCH.
