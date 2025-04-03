# HPC setup taken from
# https://github.com/aws-samples/awsome-distributed-training/blob/0ef9be61cd8cbc0b9744fce51c2a388e1a95877c/3.test_cases/18.deepspeed/0.deepspeed.dockerfile

# uses pytorch 2.6.0a0+df5bbc09d1 and NCCL 2.22.3
FROM nvcr.io/nvidia/pytorch:24.11-py3
ENV DEBIAN_FRONTEND=noninteractive
ENV NCCL_VERSION=2.22.3

# The three must-be-built packages.
# Efa-installer required for nccl to avoid libfabric NCCL error.
# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html
# latest versions of installer listed here at bottom
# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-verify.html
ENV EFA_INSTALLER_VERSION=1.37.0

# https://github.com/aws/aws-ofi-nccl/releases
# The 1.13.x release series supports NCCL 2.23.4-1 while maintaining backward compatibility with older NCCL versions (NCCL v2.17.1 and later).
ENV AWS_OFI_NCCL_VERSION=1.13.1-aws
ENV NCCL_TESTS_VERSION=master

RUN apt-get update -y
RUN apt-get remove -y --allow-change-held-packages \
                      libmlx5-1 ibverbs-utils libibverbs-dev libibverbs1

# We noticed that since 23.09, we can't just delete the whole /opt/hpcx/, otherwise `import torch`
# complains about missing libuc?.so.
RUN rm -rf /opt/hpcx/ompi \
    && rm -rf /usr/local/mpi \
    && rm -rf /opt/hpcx/nccl_rdma_sharp_plugin \
    && ldconfig
ENV OPAL_PREFIX=
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    git \
    gcc \
    vim \
    kmod \
    openssh-client \
    openssh-server \
    build-essential \
    curl \
    autoconf \
    libtool \
    gdb \
    automake \
    cmake \
    apt-utils \
    libhwloc-dev \
    aptitude && \
    DEBIAN_FRONTEND=noninteractive apt autoremove -y

# EFA
RUN apt-get update && \
    cd /tmp && \
    curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz  && \
    tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    cd aws-efa-installer && \
    # ONLY add `--skip-kmod`, `--no-verify` and `--skip-limit-conf` flags to container image.
    # Those three flags must NOT be used on the host.
    #
    # Explanations:
    # - to build EFA in the Dockerfile, we added --skip-kmod and --no-verify. Without these flags,
    #   the Dockerfile will fail to build. If installing EFA on the host and not in a container,
    #   please remove these flags.
    # - The --skip-limit-conf can be retained in Dockerfile, but it's redundant as the host already
    #   has these limits set by efa_installer.
    ./efa_installer.sh -y -g -d --skip-kmod --no-verify --skip-limit-conf && \
    ldconfig && \
    rm -rf /tmp/aws-efa-installer /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH

# NCCL EFA Plugin
RUN mkdir -p /tmp && \
    cd /tmp && \
    curl -LO https://github.com/aws/aws-ofi-nccl/archive/refs/tags/v${AWS_OFI_NCCL_VERSION}.tar.gz && \
    tar -xzf /tmp/v${AWS_OFI_NCCL_VERSION}.tar.gz && \
    rm /tmp/v${AWS_OFI_NCCL_VERSION}.tar.gz && \
    mv aws-ofi-nccl-${AWS_OFI_NCCL_VERSION} aws-ofi-nccl && \
    cd /tmp/aws-ofi-nccl && \
    ./autogen.sh && \
    ./configure --prefix=/opt/amazon/efa \
        --with-libfabric=/opt/amazon/efa \
        --with-cuda=/usr/local/cuda \
        --enable-platform-aws \
        --with-mpi=/opt/amazon/openmpi && \
    make -j$(nproc) install && \
    rm -rf /tmp/aws-ofi/nccl

# Do this to minimize the ld path env vars that users need to define when running this image.
RUN echo "/usr/local/lib"      >> /etc/ld.so.conf.d/local.conf && \
    echo "/opt/amazon/openmpi/lib" >> /etc/ld.so.conf.d/efa.conf && \
    ldconfig

ENV OMPI_MCA_pml=^cm,ucx            \
    OMPI_MCA_btl=tcp,self           \
    OMPI_MCA_btl_tcp_if_exclude=lo,docker0 \
    OPAL_PREFIX=/opt/amazon/openmpi \
    # https://discuss.pytorch.org/t/nccl-network-is-unreachable-connection-refused-when-initializing-ddp/137352
    # https://github.com/pytorch/pytorch/issues/68893
    NCCL_SOCKET_IFNAME=^docker,lo

ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# NCCL-tests: always good to include this as a diagnostic tool.
RUN git clone https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
    && cd /opt/nccl-tests \
    && git checkout ${NCCL_TESTS_VERSION} \
    && make MPI=1 \
    MPI_HOME=/opt/amazon/openmpi \
    CUDA_HOME=/usr/local/cuda \
    NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_80,code=sm_80"


##############################################
## JPT Dependencies

RUN pip install pyarrow
RUN pip install fire==0.5.0
RUN pip install transformers
RUN pip install ibm-fms
RUN pip install datasets
RUN pip install tiktoken==0.6.0
RUN pip install wandb
RUN pip install boto3
RUN pip install mosaicml-streaming
RUN pip install accelerate==0.30.0
RUN pip install torch_tb_profiler

# data requirements
RUN pip install dolma
RUN pip install smart_open
RUN pip install msgspec

# install eval harness
WORKDIR /tmp/unique_for_eval_harness
RUN git clone https://github.com/EleutherAI/lm-evaluation-harness
WORKDIR /tmp/unique_for_eval_harness/lm-evaluation-harness
RUN pip install -e .
WORKDIR /workspace
# install dependency for ifeval
RUN pip install immutabledict
RUN pip install langdetect

# install fastchat for MT eval
WORKDIR /tmp/unique_for_fastchat
RUN git clone https://github.com/jasonkrone/FastChat.git
WORKDIR /tmp/unique_for_fastchat/FastChat
RUN pip install -e ".[model_worker,llm_judge]"
WORKDIR /workspace
RUN pip install shortuuid

# install specific version of antlr4 for minerva_math dataset
RUN pip install antlr4-python3-runtime==4.11

# specific version of hf hub that prevents race condition with model downloads
# https://github.com/huggingface/transformers/issues/31019
RUN pip install huggingface_hub==0.24.7

# install alpaca eval
RUN pip install alpaca-eval 

# install ray for HPO
RUN pip install ray[default]==2.40.0
RUN pip install optuna