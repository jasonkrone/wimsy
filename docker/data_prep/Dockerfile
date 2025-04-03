# copied from dolma Dockerfile
# https://github.com/allenai/dolma/blob/c029e94445290d79ae4ee293e5bf5645b5cacfa9/.devcontainer/Dockerfile
FROM python:3.10-bullseye
SHELL ["/bin/bash", "-c"]

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    vim \
    build-essential \
    libtool \
    autoconf \
    unzip \
    wget \
    pkg-config \
    libssl-dev

RUN if [ $(uname -m) = "aarch64" ] ; then \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"; \
    else \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"; \
    fi

RUN unzip awscliv2.zip && ./aws/install

RUN wget https://github.com/peak/s5cmd/releases/download/v2.0.0/s5cmd_2.0.0_Linux-64bit.tar.gz \
    && tar -xzvf s5cmd_2.0.0_Linux-64bit.tar.gz \
    && mv s5cmd /usr/local/bin/ \
    && rm s5cmd_2.0.0_Linux-64bit.tar.gz

WORKDIR /workspaces

##############################################
#          data prep dependencies            #
##############################################

RUN pip install --upgrade pip
RUN pip install requests==2.0.0
RUN pip install msgspec
RUN pip install transformers
RUN pip install datasets
RUN pip install tiktoken==0.6.0
RUN pip install wandb
RUN pip install boto3
RUN pip install mosaicml-streaming
RUN pip install dolma

# install eval harness requirements
RUN pip install accelerate>=0.26.0
RUN pip install evaluate
RUN pip install datasets>=2.16.0
RUN pip install evaluate>=0.4.0
RUN pip install jsonlines
RUN pip install numexpr
RUN pip install peft>=0.2.0
RUN pip install pybind11>=2.6.2
RUN pip install pytablewriter
RUN pip install rouge-score>=0.0.4
RUN pip install sacrebleu>=1.5.0
RUN pip install scikit-learn>=0.24.1
RUN pip install sqlitedict
RUN pip install torch>=1.8
RUN pip install tqdm-multiprocess
RUN pip install transformers>=4.1
RUN pip install zstandard
RUN pip install dill
RUN pip install word2number
RUN pip install more_itertools

# install eval harness
WORKDIR /tmp/unique_for_eval_harness
RUN git clone https://github.com/EleutherAI/lm-evaluation-harness
WORKDIR /tmp/unique_for_eval_harness/lm-evaluation-harness
RUN pip install -e .
WORKDIR /workspace

# install dependency for ifeval
RUN pip install immutabledict
RUN pip install langdetect

RUN pip install pytest
RUN pip install elasticsearch
RUN pip install jsonpath_ng

RUN pip install jupyterlab