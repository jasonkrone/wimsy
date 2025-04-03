"""
Small modification on original MT bench gen_model_answers.py to call my get_model_answers fn
"""
import sys
import json
import os
import random
import time
import math

import ray
import torch
import shortuuid
from tqdm import tqdm

sys.path.append("./FastChat")
from fastchat.llm_judge.gen_model_answer import reorg_answer_file
from fastchat.llm_judge.common import load_questions, temperature_config



def decode_and_postprocess_output(output_ids, tokenizer, stop_str, stop_token_ids):
    # be consistent with the template's stop_token_ids
   
    if stop_token_ids:
        stop_token_ids_index = [
            i
            for i, id in enumerate(output_ids)
            if id in stop_token_ids
        ]
        if len(stop_token_ids_index) > 0:
            output_ids = output_ids[: stop_token_ids_index[0]]

    output = tokenizer.decode(output_ids)

    if stop_str and isinstance(stop_str, list):
        stop_str_indices = sorted(
            [
                output.find(stop_str)
                for stop_str in stop_str
                if output.find(stop_str) > 0
            ]
        )
        if len(stop_str_indices) > 0:
            output = output[: stop_str_indices[0]]
    elif stop_str and output.find(stop_str) > 0:
        output = output[: output.find(stop_str)]

    for special_token in tokenizer.special_tokens_map.values():
        if isinstance(special_token, list):
            for special_tok in special_token:
                output = output.replace(special_tok, "")
        else:
            output = output.replace(special_token, "")
    output = output.strip()
    return output


@torch.inference_mode()
def get_model_answers(
    get_model_fn,
    model_id,
    questions,
    max_new_token,
    num_choices,
):
    answer_list = []
    device = torch.cuda.current_device()
    model = get_model_fn(device=f"cuda:{device}")
    tokenizer = model.tokenizer
    stop_str = tokenizer.eos_token
    stop_token_ids = None 
    if stop_str is not None:
        stop_token_ids = tokenizer.convert_tokens_to_ids([stop_str])

    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)

            turns = []
            conversation = []

            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conversation.append({"role": "user", "content": qs})
                prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
                input_ids = input_ids.to(device)
                max_length = input_ids.shape[1] + max_new_token

                try:
                    do_sample = False if temperature < 1e-4 else True
                    output_ids = model._model_generate(
                        context=input_ids,
                        max_length=max_length,
                        stop=[model.tok_decode(model.eot_token_id, skip_special_tokens=False)],
                        temperature=temperature,
                        do_sample=do_sample,
                    )
                    output_ids = output_ids[0][len(input_ids[0]) :]
                    output = decode_and_postprocess_output(output_ids, tokenizer, stop_str, stop_token_ids)

                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    print("error:", e)
                    output = "ERROR"

                conversation.append({"role": "assistant", "content": output})
                turns.append(output)

            choices.append({"index": i, "turns": turns})

        ans_json = {
            "question_id": question["question_id"],
            "answer_id": shortuuid.uuid(),
            "model_id": model_id,
            "choices": choices,
            "tstamp": time.time(),
        }
        answer_list.append(json.dumps(ans_json) + "\n")
    return answer_list

    # save answers to jsonl


def write_mt_bench_answer_file(
    get_model_fn,
    model_id, 
    device, 
    questions_path, 
    answers_path, 
    max_new_tokens,
    num_choices,
    limit,
    questions_begin=None, 
    questions_end=None,
    world_size=1,
):

    use_ray = world_size > 1
    if use_ray:
        ray.init(num_gpus=world_size)

    questions = load_questions(questions_path, questions_begin, questions_end)
    questions = questions[:limit]
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    get_answers_fn = get_model_answers
    if use_ray:
        get_answers_fn = ray.remote(num_gpus=1)(get_model_answers).remote
        
    answers = []
    chunk_size = len(questions) // world_size
    for i in range(0, len(questions), chunk_size):
        print(f"num questions: {len(questions)} start: {i} end: {i+chunk_size}")
        answers.append(
            get_answers_fn(
                get_model_fn=get_model_fn,
                model_id=model_id,
                questions=questions[i:i+chunk_size],
                max_new_token=max_new_tokens,
                num_choices=num_choices,
            )
        )
    
    if use_ray:
        answers = [ray.get(a_list) for a_list in answers]

    answers = [a for a_list in answers for a in a_list]
    os.makedirs(os.path.dirname(answers_path), exist_ok=True)
    with open(os.path.expanduser(answers_path), "w") as f:
        f.writelines(answers)

    reorg_answer_file(answers_path)

    if use_ray:
        ray.shutdown()

