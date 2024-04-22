FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt upgrade -y
RUN apt-get update && apt-get install -y --no-install-recommends \
            apt-utils \
            build-essential \
            pkg-config \
            rsync \
            software-properties-common \
            unzip \
            zip \
            zlib1g-dev \
            wget \
            curl \
            git \
            git-lfs \
            vim-gtk \
            screen \
            virtualenv \
            tzdata \
            && \
            apt-get clean && \
            rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH /usr/local/cuda/bin:$PATH
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN git lfs install

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
	bash Miniconda3-latest-Linux-x86_64.sh -b && \
	rm Miniconda3-latest-Linux-x86_64.sh

SHELL ["/root/miniconda3/bin/conda", "run", "-n", "base", "/bin/bash", "-c"]
RUN conda create -n main python=3 -y
SHELL ["/root/miniconda3/bin/conda", "run", "-n", "main", "/bin/bash", "-c"]
RUN conda init bash

RUN mkdir /app/
ADD requirements.txt /app/
RUN pip --no-cache-dir install -r /app/requirements.txt
ADD . /app/
RUN pip --no-cache-dir install -e /app/

ENTRYPOINT ["/bin/bash"]
