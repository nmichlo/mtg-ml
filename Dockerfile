
# chose the base pytorch image:
# -- chose a pytorch version from: https://hub.docker.com/r/pytorch/pytorch/tags
# -- this is probably overkill and may use a lot storage space because
#    we overwrite the conda install... but its easier... pytorch may not
#    have some of the performance benefits of the original container but we
#    need the higher python version.
ARG BASE="pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime"

# default
FROM ${BASE} as pytorch

# https://pytorch.org/get-started/locally/ -- [ stable | linux | conda | python | cuda10.2 ]
RUN conda create --name torch397 -c pytorch \
      python=3.9.7 \
      pytorch=1.10.0 \
      torchvision=0.11.1 \
      torchaudio=0.10.0 \
      cudatoolkit=10.2

# AT THIS POINT IN TIME:
# LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
# PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# -- we need to remove the reference to the original conda
ENV LD_LIBRARY_PATH="/opt/conda/envs/torch397/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
ENV PATH="/opt/conda/envs/torch397/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# install the requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# make sure we can see the package
# -- usually we want to override these by specifying a volume to replace them, this enables easy development
ENV PYTHONPATH="/workspace"
COPY mtg_ml mtg_ml
COPY experiment experiment
