#!/bin/bash


function evaluate() {

    cd /sky_workdir

    nproc_per_node=${SKYPILOT_NUM_GPUS_PER_NODE}

    source keys.env; LOGLEVEL=INFO PYTHONPATH=/sky_workdir/OLMo:$PYTHONPATH accelerate launch \
        --num_processes $nproc_per_node \
        --num_machines 1 \
        --mixed_precision no \
        --dynamo_backend no \
        -m evaluation.eval_wrapper --config $1 
}



function start_enroot() {
  # /tmp/.cache
  enroot start \
    --rw \
    --mount ~/sky_workdir:/sky_workdir \
    --mount /tmp:/tmp \
    --mount /output:/output \
    --mount /artifacts:/artifacts \
    --env TRANSFORMERS_CACHE=~/.cache \
    alpaca

    #hf_hub
    #train_env 
}


function run_llama() {
    export PYTHONPATH=/sky_workdir/OLMo:$PYTHONPATH
    source keys.env ; python -m temp_llama2.temp_llama2
}


function run_elm() {
    export PYTHONPATH=/sky_workdir/OLMo:$PYTHONPATH
    source keys.env ; CUDA_LAUNCH_BLOCKING=1 python -m openelm.debug_elm
}


function run_dev_eval() {

    export PYTHONPATH=/sky_workdir/OLMo:$PYTHONPATH
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

    source keys.env; accelerate launch \
        --num_processes 8 \
        --num_machines 1 \
        --mixed_precision no \
        --dynamo_backend no \
        -m evaluation.eval_wrapper --config ./configs/eval/dev_eval/baselines_mc_dev_eval.yaml

}


function run_finetune() {
    # TODO: don't need this given we can download on multi-process now
    export PYTHONPATH=/sky_workdir/OLMo:$PYTHONPATH

    #python -c "from transformers import AutoModelForCausalLM; from hf_olmo import OLMoForCausalLM; AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path='$repo_id', trust_remote_code=True)"

    #source keys.env; CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 TOKENIZERS_PARALLELISM=false torchrun \
    #    --nproc_per_node 7 \
    #    --standalone -m training.finetune_hf_models \
    #    --finetune_config ./configs/eval/dev_eval/finetune_baselines.yaml 


    source keys.env; CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 TOKENIZERS_PARALLELISM=false torchrun \
        --nproc_per_node 7 \
        --standalone -m training.train \
        --finetune_config ./configs/eval/dev_eval/wimsy_1b/finetune_baselines.yaml 

}


function run_agi_eval_pythia() {
    task=agieval_nous
    #model_args="pretrained=EleutherAI/pythia-1b-deduped,revision=step23000,dtype=float32"
    model_args="pretrained=EleutherAI/pythia-6.9b-deduped,dtype=float32"
    out="/output/baselines_07_08_24"

    source keys.env; python -m evaluation.eval \
        --model hf \
        --model_args $model_args \
        --tasks $task \
        --output_path $out \
        --device cuda:0 \
        --batch_size 3 \
        --log_samples
}

function random_baseline() {

    export PYTHONPATH=/sky_workdir/OLMo:$PYTHONPATH
    source keys.env; CUDA_VISIBLE_DEVICES=7 accelerate launch \
        --num_processes 1 \
        --num_machines 1 \
        --mixed_precision no \
        --dynamo_backend no \
        -m evaluation.eval_wrapper --config ./configs/eval/random_4096_ctx.yaml

    #source keys.env; RANK=1 LOCAL_RANK=1 python -m evaluation.eval_wrapper --config ./configs/eval/random_4096_ctx.yaml
}


function test_olmo() {
    export PYTHONPATH=/sky_workdir/OLMo:$PYTHONPATH
    python -m temp
}


function run_test_eval() {

    # prevent punkt download issue during eval

    export PYTHONPATH=/sky_workdir/OLMo:$PYTHONPATH
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

    # random baseline
    #source keys.env; accelerate launch \
    #    --num_processes 1 \
    #    --num_machines 1 \
    #    --mixed_precision no \
    #    --dynamo_backend no \
    #    -m evaluation.eval_wrapper --config ./configs/eval/random_4096_ctx.yaml

    #source keys.env; python -m evaluation.eval_wrapper --config ./configs/eval/random_4096_ctx.yaml

    #export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

    # 4096 ctx pre-train baselines
    #source keys.env; accelerate launch \
    #    --num_processes 8 \
    #    --num_machines 1 \
    #    --mixed_precision no \
    #    --dynamo_backend no \
    #    -m evaluation.eval_wrapper --config ./configs/eval/test_eval/pretrain_baselines_4096_ctx_test_eval.yaml


    # 4096 ctx post-train baselines
    source keys.env; accelerate launch \
        --num_processes 1 \
        --num_machines 1 \
        --mixed_precision no \
        --dynamo_backend no \
        -m evaluation.eval_wrapper --config ./configs/eval/test_eval/posttrain_baselines_4096_ctx_test_eval.yaml

    # 2048 ctx pre-train baselines
    #source keys.env; accelerate launch \
    #    --num_processes 7 \
    #    --num_machines 1 \
    #    --mixed_precision no \
    #    --dynamo_backend no \
    #    -m evaluation.eval_wrapper --config ./configs/eval/test_eval/pretrain_baselines_2048_ctx_test_eval.yaml


    # 2048 ctx post-train baselines
    #source keys.env; accelerate launch \
    #    --num_processes 1 \
    #    --num_machines 1 \
    #    --mixed_precision no \
    #    --dynamo_backend no \
    #    -m evaluation.eval_wrapper --config ./configs/eval/test_eval/posttrain_baselines_2048_ctx_test_eval.yaml


    #source keys.env; CUDA_VISIBLE_DEVICES=7 python -m evaluation.eval_wrapper --config ./configs/eval/baselines_2048_ctx.yaml
}



function run_alpaca_eval() {
    export PYTHONPATH=/sky_workdir/OLMo:$PYTHONPATH
    export PYTHONPATH=/sky_workdir/lm-evaluation-harness:$PYTHONPATH
    source keys.env;  python -m tasks.alpaca_eval.alpaca_eval --config ./configs/eval/test_eval/alpaca_eval.yaml
}



function run_mt_eval() {
    export PYTHONPATH=/sky_workdir/OLMo:$PYTHONPATH
    export PYTHONPATH=/sky_workdir/lm-evaluation-harness:$PYTHONPATH
    CUDA_VISIBLE_DEVICES=7
    source keys.env;  python -m tasks.mt_bench.eval_mt_bench --config ./configs/eval/test_eval/mt_bench.yaml
}


function replicate() {
    # llama3
    #source keys.env; accelerate launch \
    #    --num_processes 8 \
    #    --num_machines 1 \
    #    --mixed_precision no \
    #    --dynamo_backend no \
    #    -m evaluation.eval_wrapper --config ./configs/eval/replicate_llama3.yaml

    # llama2
    source keys.env; accelerate launch \
        --num_processes 8 \
        --num_machines 1 \
        --mixed_precision no \
        --dynamo_backend no \
        -m evaluation.eval_wrapper --config ./configs/eval/replicate_llama2.yaml

    # llama3 instruct
    #source keys.env; accelerate launch \
    #    --num_processes 8 \
    #    --num_machines 1 \
    #    --mixed_precision no \
    #    --dynamo_backend no \
    #    -m evaluation.eval_wrapper --config ./configs/eval/replicate_llama3_instruct.yaml

    # gemma2
    #source keys.env; accelerate launch \
    #    --num_processes 8 \
    #    --num_machines 1 \
    #    --mixed_precision no \
    #    --dynamo_backend no \
    #    -m evaluation.eval_wrapper --config ./configs/eval/replicate_gemma.yaml

}



function run_llama_baseline() {

    #source keys.env; python -m evaluation.eval_wrapper --config ./configs/eval/baselines/llama2_7b.yaml

    source keys.env; accelerate launch \
        --num_processes 8 \
        --num_machines 1 \
        --mixed_precision no \
        --dynamo_backend no \
        -m evaluation.eval_wrapper --config ./configs/eval/baselines/llama2_7b.yaml
}

function csqa() {
    src=./lm-evaluation-harness
    cd $src

    model_args="pretrained=EleutherAI/pythia-160m"
    python -m lm_eval \
        --model hf \
        --model_args $model_args \
        --tasks commonsense_qa_cot_fewshot \
        --limit 10 \
        --batch_size 8 \
        --device cuda:0 \
        --log_samples \
        --output_path /output/csqa_out
    cd ..
}


function run_pythia_1b() {
    source keys.env; accelerate launch \
        --num_processes 8 \
        --num_machines 1 \
        --mixed_precision no \
        --dynamo_backend no \
        -m evaluation.eval_wrapper --config ./configs/eval/baselines/pythia1dot4b.yaml
}




function run_llama_human_eval() {

    #out="/output/baselines_07_10_24"
    ##model_args="pretrained=meta-llama/CodeLlama-7b-hf"
    #model_args="pretrained=meta-llama/Llama-2-7b-hf,dtype=bfloat16"
    #task=humaneval_greedy
    ##task=humaneval

    source keys.env; python -m evaluation.eval_wrapper --config ./configs/eval/baselines/code_llama_7b.yaml

    #source keys.env; HF_ALLOW_CODE_EVAL=1 python -m evaluation.eval \
    #    --model hf \
    #    --model_args $model_args \
    #    --tasks $task \
    #    --output_path $out \
    #    --device cuda:0 \
    #    --batch_size 3 \
    #    --log_samples
}


function run_human_eval() {
    src=./bigcode-evaluation-harness
    model_args="meta-llama/Llama-2-7b-hf"
    #model=codeparrot/codeparrot-small

    # TODO: temporary limit, arbitrary max_len_generation

    cd $src

    accelerate launch main.py \
      --model $model_args \
      --max_length_generation 512 \
      --tasks humaneval \
      --temperature 0.2 \
      --n_samples 200 \
      --batch_size 10 \
      --limit 10 \
      --allow_code_execution

    cd ..
}


function run_eval_code_llama() {
    source keys.env; torchrun --nproc_per_node 1 --standalone -m evaluation.eval_wrapper --config ./configs/eval/baselines/code_llama_7b.yaml
}


function run_eval_mine() {
    # TODO: i should have a single script called run-baselines that gives you all the baseline results
    #source keys.env; torchrun --nproc_per_node 1 --standalone -m evaluation.eval_wrapper --config ./configs/eval/baselines/random.yaml
    #source keys.env; python -m evaluation.eval_wrapper --config ./configs/eval/baselines/random.yaml

    gpus=8
    #gpus=1

    if [ "$gpus" -eq 8 ]; then
        source keys.env; accelerate launch \
            --num_processes 8 \
            --num_machines 1 \
            --mixed_precision no \
            --dynamo_backend no \
            -m evaluation.eval_wrapper --config ./configs/eval/baselines/random.yaml
    else
        echo "running single node version"
        source keys.env; torchrun \
            --nproc_per_node $gpus \
            --standalone \
            -m evaluation.eval_wrapper --config ./configs/eval/baselines/random.yaml
    fi
}



function evaluate_all_tasks() {
    # $1: model args
    # $2: output base path
    # $3: device
    # $4: config_dir
    # $5: datasets (optional) if not given defaults to all datasets

    tasks=(
        "arc_easy"
        "arc_challenge"
        "boolq"
        "gsm8k_yaml"
        "hellaswag"
        "lambada_openai"
        "openbookqa"
        "piqa"
        "squadv2"
        "triviaqa_1shot"
        "winogrande"
        "mmlu_5shot"
        "mmlu_flan_cot_fewshot"
    )

    # if datasets are not given, use all datasets
    if [ -z "$5" ]; then
        task_str=$(printf "%s," "${tasks[@]}")
        task_str=${task_str%,}
    else
        task_str="$5"
    fi

    echo "$task_str"

    python eval.py \
        --model hf \
        --model_args $1 \
        --tasks $task_str \
        --output_path $2 \
        --device $3 \
        --batch_size auto \
        --include_path $4 \
        --log_samples
}


function evaluate_all_tasks_gptj() {
    # $1: model args
    # $2: output base path
    # $3: device
    # $4: config_dir
    # $5: datasets (optional) if not given defaults to all datasets

    tasks=(
        "arc_easy"
        "arc_challenge"
        "boolq"
        "gsm8k_yaml"
        "hellaswag"
        "lambada_openai"
        "openbookqa"
        "piqa"
        "squadv2"
        "triviaqa_1shot"
        "winogrande"
        "mmlu_5shot"
        "mmlu_flan_cot_fewshot"
    )

    # if datasets are not given, use all datasets
    if [ -z "$5" ]; then
        task_str=$(printf "%s," "${tasks[@]}")
        task_str=${task_str%,}
    else
        task_str="$5"
    fi

    echo "$task_str"

    python eval.py \
        --model hf \
        --model_args $1 \
        --tasks $task_str \
        --output_path $2 \
        --device $3 \
        --batch_size auto \
        --include_path $4
}


function evaluate_all_tasks_limit() {
    # $1: model args
    # $2: output base path
    # $3: device
    # $4: config_dir
    # $5: datasets (optional) if not given defaults to all datasets

    tasks=(
        "arc_easy"
        "arc_challenge"
        "boolq"
        "gsm8k_yaml"
        "hellaswag"
        "lambada_openai"
        "openbookqa"
        "piqa"
        "squadv2"
        "triviaqa_1shot"
        "winogrande"
        "mmlu_5shot"
        "mmlu_flan_cot_fewshot"
    )

    # if datasets are not given, use all datasets
    if [ -z "$5" ]; then
        task_str=$(printf "%s," "${tasks[@]}")
        task_str=${task_str%,}
    else
        task_str="$5"
    fi

    echo "$task_str"

    python eval.py \
        --model hf \
        --model_args $1 \
        --tasks $task_str \
        --output_path $2 \
        --device $3 \
        --batch_size 4 \
        --include_path $4 \
        --log_samples \
        --limit $6
}



# this allows the script to be easily run with enroot
if [[ $# -gt 0 ]]; then
    "$@"
else
    echo "No arguments were provided."
fi





