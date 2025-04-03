#!/bin/bash

SRC_DIR=/sky_workdir



function finetune_hf() {

    cd $SRC_DIR

    keys_path=$SRC_DIR/keys.env
    nproc_per_node=${SKYPILOT_NUM_GPUS_PER_NODE}

    source $keys_path; TOKENIZERS_PARALLELISM=false PYTHONPATH=/sky_workdir/OLMo:$PYTHONPATH torchrun \
        --nproc_per_node $nproc_per_node \
        --standalone -m training.finetune_hf_models \
        --config $1 
}



function finetune_wimsy() {

    cd $SRC_DIR

    keys_path=$SRC_DIR/keys.env
    nproc_per_node=${SKYPILOT_NUM_GPUS_PER_NODE}

    find . -name "*.pyc" -delete

    source $keys_path; torchrun \
        --nproc_per_node $nproc_per_node \
        --standalone -m training.train \
        --config $1 
}



function run_ai2_tulu_v3_dpo() {
    # TODO
    cd ./open-instruct
    keys_path=~/sky_workdir/keys.env

    MACHINE_RANK=0
    MAIN_PROCESS_IP=`echo "$SKYPILOT_NODE_IPS" | head -n1`
    NUM_MACHINES=1
    NUM_PROCESSES=8

    source $keys_path; accelerate launch \
    --mixed_precision bf16 \
    --num_machines $NUM_MACHINES \
    --num_processes $NUM_PROCESSES \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf open_instruct/dpo_tune_cache.py \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-SFT \
    --use_flash_attn \
    --tokenizer_name allenai/Llama-3.1-Tulu-3-8B-SFT \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-07 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --output_dir /output/02_15_25_ai2_source_code_llama3dot1_8b_dpo \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --model_revision main \
    --gradient_checkpointing \
    --dataset_mixer_list allenai/llama-3.1-tulu-3-8b-preference-mixture 1.0 \
    --use_slow_tokenizer \
    --use_lora False \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5 \
    --checkpointing_steps 1000 \
    --exp_name 02_15_25_ai2_source_code_llama3dot1_8b_dpo

}




function run_ai2_tulu_v3_sft() {

    cd ./open-instruct

    MACHINE_RANK=0
    MAIN_PROCESS_IP=`echo "$SKYPILOT_NODE_IPS" | head -n1`
    NUM_MACHINES=1
    NUM_PROCESSES=8
    PER_DEVICE_TRAIN_BATCH_SIZE=1
    GRADIENT_ACCUMULATION_STEPS=16

    keys_path=~/sky_workdir/keys.env

    source $keys_path; accelerate launch \
        --mixed_precision bf16 \
        --num_machines $NUM_MACHINES \
        --num_processes $NUM_PROCESSES \
        --machine_rank $MACHINE_RANK \
        --main_process_ip $MAIN_PROCESS_IP \
        --main_process_port 29400 \
        --use_deepspeed \
        --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
        --deepspeed_multinode_launcher standard open_instruct/finetune.py \
        --model_name_or_path meta-llama/Llama-3.1-8B \
        --tokenizer_name meta-llama/Llama-3.1-8B \
        --use_slow_tokenizer \
        --use_flash_attn \
        --max_seq_length 4096 \
        --preprocessing_num_workers 128 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --learning_rate 5e-06 \
        --lr_scheduler_type linear \
        --warmup_ratio 0.03 \
        --weight_decay 0.0 \
        --num_train_epochs 2 \
        --output_dir /output/02_15_25_ai2_source_code_llama3dot1_8b_sft \
        --with_tracking \
        --report_to wandb \
        --logging_steps 1 \
        --reduce_loss sum \
        --model_revision main \
        --dataset_mixer_list allenai/tulu-3-sft-mixture 1.0 \
        --checkpointing_steps 1000 \
        --dataset_mix_dir /output/02_15_25_ai2_source_code_llama3dot1_8b_sft \
        --exp_name tulu-3-8b-sft-ai2-02-15-25 \
        --seed 123

        #--checkpointing_steps epoch \
}

function train_expand_segments() {
    #1: config path
    #export NCCL_DEBUG=INFO
    export FI_EFA_FORK_SAFE=1
    export FI_LOG_LEVEL=1 # error
    export FI_PROVIDER=efa
    export FI_EFA_USE_HUGE_PAGE=0
    ## Switching SYNC_MEMOPS to zero can boost throughput with FSDP
    ## Disables CU_POINTER_ATTRIBUTE_SYNC_MEMOPS
    ## Reduces memory synchronizations
    ## https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html
    export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
    # https://discuss.pytorch.org/t/nccl-network-is-unreachable-connection-refused-when-initializing-ddp/137352
    # https://github.com/pytorch/pytorch/issues/68893
    export NCCL_SOCKET_IFNAME=en
    export NCCL_ASYNC_ERROR_HANDLING=1

    #can help improve the all-reduce times over the replication process group for some cluster setups.
    export NCCL_CROSS_NIC=1 

    num_nodes=${SKYPILOT_NUM_NODES}
    master_addr=`echo "$SKYPILOT_NODE_IPS" | head -n1`
    nproc_per_node=${SKYPILOT_NUM_GPUS_PER_NODE}
    node_rank=${SKYPILOT_NODE_RANK}

    echo $SKYPILOT_NODE_IPS
    echo "num_nodes: $num_nodes"
    echo "master_addr: $master_addr"
    echo "nproc_per_node: $nproc_per_node"
    echo "node_rank: $node_rank"

    cd $SRC_DIR

    keys_path=$SRC_DIR/keys.env

    #export TORCHDYNAMO_VERBOSE=1
    #export TORCH_LOGS="+dynamo"

    find . -name "*.pyc" -delete

    source $keys_path ; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LOG_LEVEL=INFO torchrun \
     --nnodes=$num_nodes \
     --nproc_per_node=$nproc_per_node \
     --master_addr=$master_addr \
     --master_port="1234" \
     --node_rank=$node_rank \
     -m training.train --config $1
}


function train() {
    #1: config path
    #export NCCL_DEBUG=INFO
    export FI_EFA_FORK_SAFE=1
    export FI_LOG_LEVEL=1 # error
    export FI_PROVIDER=efa
    export FI_EFA_USE_HUGE_PAGE=0
    ## Switching SYNC_MEMOPS to zero can boost throughput with FSDP
    ## Disables CU_POINTER_ATTRIBUTE_SYNC_MEMOPS
    ## Reduces memory synchronizations
    ## https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html
    export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
    # https://discuss.pytorch.org/t/nccl-network-is-unreachable-connection-refused-when-initializing-ddp/137352
    # https://github.com/pytorch/pytorch/issues/68893
    export NCCL_SOCKET_IFNAME=en
    export NCCL_ASYNC_ERROR_HANDLING=1

    #can help improve the all-reduce times over the replication process group for some cluster setups.
    export NCCL_CROSS_NIC=1 

    num_nodes=${SKYPILOT_NUM_NODES}
    master_addr=`echo "$SKYPILOT_NODE_IPS" | head -n1`
    nproc_per_node=${SKYPILOT_NUM_GPUS_PER_NODE}
    node_rank=${SKYPILOT_NODE_RANK}

    echo $SKYPILOT_NODE_IPS
    echo "num_nodes: $num_nodes"
    echo "master_addr: $master_addr"
    echo "nproc_per_node: $nproc_per_node"
    echo "node_rank: $node_rank"

    cd $SRC_DIR

    keys_path=$SRC_DIR/keys.env

    #export TORCHDYNAMO_VERBOSE=1
    #export TORCH_LOGS="+dynamo"

    find . -name "*.pyc" -delete

    source $keys_path ; LOG_LEVEL=INFO torchrun \
     --nnodes=$num_nodes \
     --nproc_per_node=$nproc_per_node \
     --master_addr=$master_addr \
     --master_port="1234" \
     --node_rank=$node_rank \
     -m training.train --config $1
}


function train_skip_ecc() {
    #1: config path
    #export NCCL_DEBUG=INFO
    export FI_EFA_FORK_SAFE=1
    export FI_LOG_LEVEL=1 # error
    export FI_PROVIDER=efa
    export FI_EFA_USE_HUGE_PAGE=0
    ## Switching SYNC_MEMOPS to zero can boost throughput with FSDP
    ## Disables CU_POINTER_ATTRIBUTE_SYNC_MEMOPS
    ## Reduces memory synchronizations
    ## https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html
    export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
    # https://discuss.pytorch.org/t/nccl-network-is-unreachable-connection-refused-when-initializing-ddp/137352
    # https://github.com/pytorch/pytorch/issues/68893
    export NCCL_SOCKET_IFNAME=en
    export NCCL_ASYNC_ERROR_HANDLING=1

    #can help improve the all-reduce times over the replication process group for some cluster setups.
    export NCCL_CROSS_NIC=1 

    num_nodes=${SKYPILOT_NUM_NODES}
    master_addr=`echo "$SKYPILOT_NODE_IPS" | head -n1`
    nproc_per_node=${SKYPILOT_NUM_GPUS_PER_NODE}
    node_rank=${SKYPILOT_NODE_RANK}

    cd $SRC_DIR

    keys_path=$SRC_DIR/keys.env

    #export TORCHDYNAMO_VERBOSE=1
    #export TORCH_LOGS="+dynamo"

    source $keys_path ; CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 torchrun \
     --nnodes=$num_nodes \
     --nproc_per_node=7 \
     --master_addr=$master_addr \
     --master_port="1234" \
     --node_rank=$node_rank \
     -m training.train --config $1
}


function train_in_phases() {
    #1: config path
    #export NCCL_DEBUG=INFO
    export FI_EFA_FORK_SAFE=1
    export FI_LOG_LEVEL=1
    export FI_PROVIDER=efa
    export FI_EFA_USE_HUGE_PAGE=0
    ## Switching SYNC_MEMOPS to zero can boost throughput with FSDP
    ## Disables CU_POINTER_ATTRIBUTE_SYNC_MEMOPS
    ## Reduces memory synchronizations
    ## https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html
    export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
    # https://discuss.pytorch.org/t/nccl-network-is-unreachable-connection-refused-when-initializing-ddp/137352
    # https://github.com/pytorch/pytorch/issues/68893
    export NCCL_SOCKET_IFNAME=en
    export NCCL_ASYNC_ERROR_HANDLING=1

    num_nodes=${SKYPILOT_NUM_NODES}
    master_addr=`echo "$SKYPILOT_NODE_IPS" | head -n1`
    nproc_per_node=${SKYPILOT_NUM_GPUS_PER_NODE}
    node_rank=${SKYPILOT_NODE_RANK}

    echo $SKYPILOT_NODE_IPS
    echo "num_nodes: $num_nodes"
    echo "master_addr: $master_addr"
    echo "nproc_per_node: $nproc_per_node"
    echo "node_rank: $node_rank"

    cd $SRC_DIR

    keys_path=$SRC_DIR/keys.env

    #export TORCHDYNAMO_VERBOSE=1
    #export TORCH_LOGS="+dynamo"

    echo "one ${1}"
    echo "two ${2}"

    source $keys_path ; torchrun \
     --nnodes=$num_nodes \
     --nproc_per_node=$nproc_per_node \
     --master_addr=$master_addr \
     --master_port="1234" \
     --node_rank=$node_rank \
     -m training.train_in_phases --configs $1 $2
}

# this allows the script to be easily run with enroot
if [[ $# -gt 0 ]]; then
    "$@"
else
    echo "No arguments were provided."
fi

