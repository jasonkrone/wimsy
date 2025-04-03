#!/bin/bash

function pull() {
    docker pull nvcr.io/nvidia/pytorch:24.04-py3
}

function build() {
    # $1: Dockerfile 
    # $2: tag
    docker build --progress=plain -t jpt:$2 -f $1 .
    # python -c "import triton ; print(triton.__version__)"
}

function mk_enroot() {
    #$1: tag (usually latest)
    enroot import -o jpt.sqsh dockerd://jpolingkrone/jpt:$1
}

function start_enroot() {

    #enroot create --name jpt jpt.sqsh
    enroot start --rw --mount .:/jpt --mount /home/ubuntu/data:/data  jpt

    # docker run -it --volume .:/fms --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  jpolingkrone/jpt:latest

    #enroot start jpt.sqsh \
    #    -rw \ # make the container root filesystem writable
    #    -r /home/ubuntu/sky_workdir/jpt \
    #    echo "hi from insde"
        #-c "enroot_config" \
        #-e "export environment vars" \
        #-r "root << i'd guess code dir" \ # << if we did this to the actual root then everything would be writable right?
        #-m "mount from the host insde the container"
}

function docker_to_enroot() {
    # $1: image, $2: tag
    docker build --progress=plain -t $1:$2 -f Dockerfile .
    enroot import -o ./$2.sqsh dockerd://$1:$2
    enroot create --name $2 $2.sqsh
    enroot start --rw --mount $2
}


function upload_img() {
    docker build --progress=plain -t jpt:0.01 -f ./docker/Dockerfile .
    docker image tag jpt:0.01 jpolingkrone/jpt:latest
    docker image push jpolingkrone/jpt:latest
}


function setup_on_awspc() {
    # : [rank52]:     from triton.compiler.compiler import triton_key
    #6: [rank52]: torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
    #6: [rank52]: ImportError: cannot import name 'triton_key' from 'triton.compiler.compiler' (/usr/local/lib/python3.10/dist-packages/triton/compiler/compiler.py)

    mkdir /fsx/apps
    # if it's already built
    docker pull jpolingkrone/jpt
    # if it's not already build
    docker pull nvcr.io/nvidia/pytorch:24.04-py3
    docker build --progress=plain -t jpt:0.01 -f ./docker/Dockerfile .
    docker image tag jpt:0.01 jpolingkrone/jpt:latest
    enroot import -o /fsx/apps/jpt.sqsh dockerd://jpolingkrone/jpt:latest
}

