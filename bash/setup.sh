#!/bin/bash


function setup_nightly() {

    #source ./miniconda3/bin/activate
    #conda create -y -p ./pt_nightly python=3.10
    dependencies=$(tr '\n' ' ' < ./requirements/nightly_conda_requirements.txt)
    echo $dependencies
    conda install $dependencies -c pytorch-nightly -c nvidia
    pip install wandb
    pip install mosaicml-streaming
    pip install ibm-fms

}


function install_tulu3_requirements() {
    pip install --upgrade pip "setuptools<70.0.0" wheel 
    # TODO, unpin setuptools when this issue in flash attention is resolved
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
    pip install packaging
    pip install flash-attn==2.7.2.post1 --no-build-isolation
    pip install -r requirements.txt
    pip install -e .
    python -m nltk.downloader punkt
}


function install_enroot() {
    # Set environment variable to noninteractive
    export DEBIAN_FRONTEND=noninteractive

    # TODO: right now this is just for ubuntu
    #sudo apt install -y curl gawk jq squashfs-tools parallel
    #sudo apt install -y fuse-overlayfs libnvidia-container-tools pigz squashfuse # optional

    sudo apt-get -o Dpkg::Options::="--force-confold" install -y curl gawk jq squashfs-tools parallel
    sudo apt-get -o Dpkg::Options::="--force-confold" install -y fuse-overlayfs libnvidia-container-tools pigz squashfuse

    # https://serverfault.com/questions/527789/how-to-automate-changed-config-files-during-apt-get-upgrade-in-ubuntu-12
    arch=$(dpkg --print-architecture)
    curl -fSsL -O https://github.com/NVIDIA/enroot/releases/download/v3.4.1/enroot_3.4.1-1_${arch}.deb
    curl -fSsL -O https://github.com/NVIDIA/enroot/releases/download/v3.4.1/enroot+caps_3.4.1-1_${arch}.deb # optional
    sudo apt install -y ./*.deb
}


function setup_enroot() {
    # $1: s3 path to sqsh file
    # $2: local path to sqsh file
    # $3: enroot env name

    # install enroot
    install_enroot
    # fix docker issue on lambda labs
    sudo usermod -aG docker $USER
    newgrp docker
    # copy sqsh file
    aws s3 cp $1 $2
    # create enroot env
    enroot create --name $3 $2
}


function activate_or_create_conda_env() {
    #$1: name of the env
    #$2: path to requirements txt if creating the env
    #$3: if we should install eval harness as well (0: no, 1: yes)
    conda deactivate
    conda activate "$1"
    if [ $? -eq 0 ]; then
        echo "conda env exists"
    else
        # Strange bug that we fix for us-east-2
        conda config --remove channels https://aws-ml-conda-ec2.s3.us-west-2.amazonaws.com
        # Setup the environment
        conda create -n "$1" python=3.11.10 -y
        conda activate "$1"
        pip install --upgrade pip setuptools wheel
        # TODO pre-install of torch is a temporary hack
        #pip install torch==2.2.1
        pip install -r "$2"
        if [ $# -eq 3 ] && [ "$3" = "1" ]; then
            install_eval_harness
        fi
    fi
}

function activate_or_create_mini_conda_env() {
    # $1: name of the env
    env_name=$1
    env_path=/home/ubuntu/$env_name
    conda_path=/home/ubuntu/miniconda3

    if [ ! -d $conda_path ]; then
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -P /home/ubuntu
        chmod +x Miniconda3-latest-Linux-x86_64.sh
        ./Miniconda3-latest-Linux-x86_64.sh -b -f -p $conda_path
    fi

    if [ ! -d $env_path ]; then
        conda create -y -p $env_path python=3.10
        conda activate $env_path
        conda install -y pytorch=2.2.0 pytorch-cuda=12.1 torchvision torchaudio transformers datasets fsspec=2023.9.2 --strict-channel-priority --override-channels -c https://aws-ml-conda.s3.us-west-2.amazonaws.com -c nvidia -c conda-forge
        conda install -y wandb huggingface_hub evaluate scipy boto3 tiktoken==0.6.0 --strict-channel-priority --override-channels -c https://aws-ml-conda.s3.us-west-2.amazonaws.com -c nvidia -c conda-forge
        pip install mosaicml-streaming
        pip install ibm-fms
    else
        conda activate $env_path
    fi
}


function install_eval_harness() {
    # git clone https://github.com/EleutherAI/lm-evaluation-harness.git
    cd ~/sky_workdir/lm-evaluation-harness
    #git fetch origin big-refactor
    #git checkout -b big-refactor origin/big-refactor
    pip install -e .
}


function clean_tmp() {
    rm -rf /tmp/*_dev
    rm -rf /tmp/*_train
    rm -rf /tmp/tmp*
    rm -rf /tmp/7a*
}


function remove_downloads_except_rank_shard() {
    # $1: data_dir
    # $2: rank_shard
    # $3: num_shards
    for i in $(seq 1 $(( $3 ))); do
        if [ $i -ne $2 ]; then
            echo "removing shard-${i}-of-$3"
            rm -rf "$1"/shard-${i}-of-$3
        fi
    done
}