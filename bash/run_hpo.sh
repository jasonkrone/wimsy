#!/bin/bash


SRC_DIR=/sky_workdir


function hpo() {

    cd $SRC_DIR

    ray_port=6379
    num_nodes=$(echo "$SKYPILOT_NODE_IPS" | wc -l)
    head_ip=$(echo "$SKYPILOT_NODE_IPS" | head -n1)

    if [ "$SKYPILOT_NODE_RANK" == "0" ]; then
        ps aux | grep ray | grep 6379 &> /dev/null || ray start --head  --disable-usage-stats --port "$ray_port" --ray-client-server-port 9999
        sleep 30
        find . -name "*.pyc" -delete
        source keys.env; TOKENIZERS_PARALLELISM=false PYTHONPATH=/sky_workdir/OLMo:$PYTHONPATH RAY_DEDUP_LOGS=1 python -m hpo.hpo --config $1
    else
        sleep 15
        ps aux | grep ray | grep 6379 &> /dev/null || ray start --address $head_ip:"$ray_port" --disable-usage-stats
    fi

}


function sweep_lr() {

    cd $SRC_DIR

    keys_path=$SRC_DIR/keys.env
    nproc_per_node=${SKYPILOT_NUM_GPUS_PER_NODE}

    source $keys_path;  torchrun \
        --nproc_per_node $nproc_per_node \
        --standalone -m hpo.sweep_lr \
        --config $1
}


# this allows the script to be easily run with enroot
if [[ $# -gt 0 ]]; then
    "$@"
else
    echo "No arguments were provided."
fi


