# FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Ubuntu setting
ARG DEBIAN_FRONTEND=noninteractive

# Installation
RUN \
    apt update -y --fix-missing &&\
    apt install -y sudo

RUN \
    sudo apt install \
    build-essential \
    clang-format \
    wget \
    software-properties-common \
    -y

RUN \
    sudo apt-get clean &&\
    sudo apt-get autoremove --purge &&\
    sudo apt-get remove python3.10 -y &&\
    sudo apt-get autoremove --purge -y

RUN wget https://bootstrap.pypa.io/get-pip.py

ARG PYTHON_SUBVERSION=3.11
ARG PYTHON_VERSION=${PYTHON_SUBVERSION}.9

# For Speed Optimization
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
RUN tar -xf Python-${PYTHON_VERSION}.tgz

RUN \
    sudo apt install \
    libffi-dev \
    zlib1g-dev \
    libssl-dev \
    libz-dev \
    libbz2-dev \
    libsqlite3-dev \
    libreadline-dev \
    libncurses5-dev \
    libncursesw5-dev \
    liblzma-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libexpat1-dev \
    libmpdec-dev \
    libmpfr-dev \
    libmpc-dev \    
    -y

RUN cd Python-${PYTHON_VERSION} \
    && ./configure --enable-optimizations \
    && make -j 48 \
    && sudo make altinstall

RUN rm *.tgz
RUN sudo update-alternatives --install /usr/bin/python python /usr/local/bin/python${PYTHON_SUBVERSION} 1

RUN python get-pip.py
RUN rm *.py

RUN pip install torch
RUN \
    pip install \
    scikit-learn \
    tqdm \
    matplotlib \
    pandas \
    scikit-build \
    scipy \ 
    numba \
    natsort \
    scikit-learn \
    scikit-learn-extra \
    torchvision \
    tensorboard 

WORKDIR /workspace
COPY . /workspace

