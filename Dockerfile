
# this release should work, but it requires a newer GPU driver... 470 or later!
# - https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
# - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html
# -
FROM nvcr.io/nvidia/pytorch:21.10-py3

# non-interactive install (tzdata dep. fix)
ENV DEBIAN_FRONTEND=noninteractive

# install cmake and build tools
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        wget \
        cmake

# install the requirements -- TODO: this might override NVIDIA specific packages and cause problems?
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY requirements-exp.txt .
RUN pip install -r requirements-exp.txt

# install horovod
# - https://horovod.readthedocs.io/en/stable/install_include.html
# - https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html
RUN HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]
