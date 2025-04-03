#!/bin/bash


function debug_run() {
    #source keys.env ; torchrun --standalone --nproc_per_node=4 -m training.train --config ./configs/debug/eval_only_wimsy_1b_phase1a_ckpt.yaml
    #source keys.env ; torchrun --standalone --nproc_per_node=8 -m training.train --config ./configs/train/wimsy/wimsy_1b/pretrain_wimsy_1b_phase2.yaml
    config=./configs/debug/mini_model_for_ckpt_debug.yaml
    find . -name "*.pyc" -delete
    rm -rf /tmp/test_optim/
    # CUDA_VISIBLE_DEVICES=0,1 
    source keys.env ; LOG_LEVEL=ERROR torchrun --standalone --nproc_per_node=8 -m debug.test_load_optim_ckpt --config $config
}


function get_debug_dev_data() {
    sudo mkdir /data
    cd /data
    source ~/sky_workdir/keys.env ; sudo aws s3 sync s3://dclm-top500b-gre-0.97-leq-1.00-ctx-4096-tokenizer-tiktoken/dev .
}

function run_convert_ckpt() {
    num_nodes=${SKYPILOT_NUM_NODES}
    master_addr=`echo "$SKYPILOT_NODE_IPS" | head -n1`
    nproc_per_node=${SKYPILOT_NUM_GPUS_PER_NODE}
    node_rank=${SKYPILOT_NODE_RANK}

    cd /sky_workdir

    torchrun \
     --nnodes=1 \
     --nproc_per_node=$nproc_per_node \
     --master_addr=$master_addr \
     --master_port="1234" \
     --node_rank=$node_rank \
     -m evaluation.convert_ckpt_for_eval --config $1

}


function enroot_bash() {

    num_nodes=`echo "$SKYPILOT_NODE_IPS" | wc -l`

    env_name=train_env

    fsx_mountpoint=/mnt/fsx

    enroot start \
        --rw \
        --mount ~/sky_workdir:/sky_workdir \
        --mount $fsx_mountpoint:/data \
        --mount /tmp:/tmp \
        --mount /output:/output \
        --env SKYPILOT_NUM_NODES=$num_nodes \
        --env SKYPILOT_NODE_IPS \
        --env SKYPILOT_NUM_GPUS_PER_NODE \
        --env SKYPILOT_NODE_RANK \
        --env TRANSFORMERS_CACHE=/tmp/.cache \
        $env_name \
        bash
}

# this allows the script to be easily run with enroot
if [[ $# -gt 0 ]]; then
    "$@"
else
    echo "No arguments were provided."
fi
