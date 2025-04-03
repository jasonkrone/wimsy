#!/bin/bash

# v5: changed min down to 0
CLUSTER_NAME=h100-x2-v1-05-21-24
CLUSTER_REGION=us-east-1
KEYPATH=~/Developer/keys/jpk-everycare-us-east-1.pem
CONFIG_PATH=./configs/awspc/apc_8xh100_east1.yaml




function mk_vpc() {
    # created that skeleton via 'aws cloudformation create-stack --generate-cli-skeleton'

    # CloudFormation template taken from https://catalog.us-east-1.prod.workshops.aws/workshops/6cbbf337-c498-4c6b-ad4f-99d22e03d8dc/en-US/01-getting-started/03-prerequisites
    # credit to Sean Smith at AWS
    aws cloudformation create-stack \
        --cli-input-json file://./configs/cloud_formation/cli-input.json \
        --template-body "$(cat ./configs/cloud_formation/vpc_and_security_group.yaml)" \
        --capabilities CAPABILITY_IAM

    # to get the outputs i.e. all the things that we just created
    # aws cloudformation describe-stacks --stack-name temp-stack-v2
}

function login_docker() {
    docker login
}

function pull_docker() {
    docker pull jpolingkrone/jpt
}

function export_vars() {

    export APPS_PATH=/apps
    export ENROOT_IMAGE=$APPS_PATH/jpt.sqsh

}


function mk_enroot() {
    enroot import -o jpt.sqsh dockerd://jpolingkrone/jpt:latest
}


function mk_cluster() {

    pcluster create-cluster \
       --cluster-name $CLUSTER_NAME \
       --cluster-configuration $CONFIG_PATH \
       --region $CLUSTER_REGION \
       --rollback-on-failure false

}

function why_failed() {
    pcluster get-cluster-stack-events --cluster-name $CLUSTER_NAME --region $CLUSTER_REGION --query 'events[?resourceStatus==`CREATE_FAILED`]'
}


function export_in_head_node() {
    export PYTHON_VERSION=3.10
    # We are using Python version 3.10 in this work. For a different Python version select the right Miniconda file from https://repo.anaconda.com/miniconda/
    export MINICONDA_INSTALLER=Miniconda3-py310_23.5.2-0-Linux-x86_64
    export TARGET_PATH=/apps/bionemo-src   # Must be a shared filesystem. This is where Nemo launcher scripts will reside.
    export DOCKER_IMAGE_NAME=bionemo
    export TAG=latest
    export ENROOT_IMAGE=/apps/${DOCKER_IMAGE_NAME}
    export DATASET_PATH=/fsx/
}


function install_nvidia_cli() {
    # I ssh'd into the head node then installed this on the compute nodes via
    # sbatch_configs -N 2 --wrap "srun nvidia_install.sh" << here's -N 2 means on 2 nodes

    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
      && \
        sudo apt-get update \
      && sudo apt-get install libnvidia-container1 \
      && sudo apt-get install libnvidia-container-tools

}


function ssh_cluster() {
    #chmod 600 ~/Developer/keys/jpk-everycare-us-east-2.pem
    #pcluster ssh -n h100-v1 --region us-east-2 -i ~/Developer/keys/jpk-everycare-us-east-2.pem

    pcluster ssh -n $CLUSTER_NAME --region $CLUSTER_REGION -i $KEYPATH
}

function ssh_for_tb() {
    ssh -i ~/Developer/keys/jpk-everycare-us-east-1.pem -L 6006:localhost:6006 ec2-user@ec2-34-204-187-99.compute-1.amazonaws.com
}


function check_cluster() {

    watch pcluster describe-cluster \
       --cluster-name $CLUSTER_NAME \
       --region $CLUSTER_REGION \
       --query clusterStatus

}

# Pull down the BioNeMo framework container
# Build an AWS-optimized image
# Run
