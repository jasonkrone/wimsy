"""
Usage:
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --mode [single|pairwise-baseline|pairwise-all]
"""
import pdb
import sys
from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
from tqdm import tqdm

sys.path.append("./FastChat")

from fastchat.llm_judge.common import (
    load_questions,
    load_judge_prompts,
    check_data,
    play_a_match_pair,
    play_a_match_single,
    Judge,
    MatchPair,
    MatchSingle,
    NEED_REF_CATS,
)


def load_model_answers(path):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    model_answers = {}
    answer = {}
    with open(path) as fin:
        for line in fin:
            line = json.loads(line)
            answer[line["question_id"]] = line
    model_id = line["model_id"]
    model_answers[model_id] = answer
    return model_answers


def make_match(
    questions,
    models,
    model_answers,
    judge,
    baseline_model,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m_1 = models[i]
            m_2 = baseline_model
            if m_1 == m_2:
                continue
            a_1 = model_answers[m_1][q_id]
            a_2 = model_answers[baseline_model][q_id]
            if ref_answers is not None:
                ref = ref_answers[judge.model_name][q_id]
                match = MatchPair(
                    dict(q),
                    m_1,
                    m_2,
                    a_1,
                    a_2,
                    judge,
                    ref_answer=ref,
                    multi_turn=multi_turn,
                )
            else:
                match = MatchPair(
                    dict(q), m_1, m_2, a_1, a_2, judge, multi_turn=multi_turn
                )
            matches.append(match)
    return matches


def make_match_all_pairs(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                q_id = q["question_id"]
                m_1 = models[i]
                m_2 = models[j]
                a_1 = model_answers[m_1][q_id]
                a_2 = model_answers[m_2][q_id]
                if ref_answers is not None:
                    ref = ref_answers[judge.model_name][q_id]
                    match = MatchPair(
                        dict(q),
                        m_1,
                        m_2,
                        a_1,
                        a_2,
                        judge,
                        ref_answer=ref,
                        multi_turn=multi_turn,
                    )
                else:
                    match = MatchPair(
                        dict(q), m_1, m_2, a_1, a_2, judge, multi_turn=multi_turn
                    )
                matches.append(match)
    return matches


def make_match_single(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m = models[i]
            a = model_answers[m][q_id]
            if ref_answers is not None:
                ref = ref_answers[judge.model_name][q_id]
                matches.append(
                    MatchSingle(
                        dict(q), m, a, judge, ref_answer=ref, multi_turn=multi_turn
                    )
                )
            else:
                matches.append(MatchSingle(dict(q), m, a, judge, multi_turn=multi_turn))
    return matches


def make_judge_pairwise(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["pair-v2"])
    judges["math"] = Judge(judge_model, judge_prompts["pair-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(
        judge_model, judge_prompts["pair-v2-multi-turn"], multi_turn=True
    )
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["pair-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


def make_judge_single(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["single-v1"])
    judges["math"] = Judge(judge_model, judge_prompts["single-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(
        judge_model, judge_prompts["single-v1-multi-turn"], multi_turn=True
    )
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["single-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


def write_mt_bench_judgements_file(
    models,
    output_file,
    question_file,
    model_answers_file,
    reference_answers_file,
    judge_file,
    judge_model,
    baseline_model,
    mode,
    num_workers,
    limit=None,
):

    questions = load_questions(question_file, None, None)
    model_answers = load_model_answers(model_answers_file)
    ref_answers = load_model_answers(reference_answers_file)
    judge_prompts = load_judge_prompts(judge_file)

    if limit:
        assert len(models) == 1
        model_name = models[0]
        question_ids = list(model_answers[model_name].keys())
        questions = [q for q in questions if q["question_id"] in question_ids]
        ref_answers[judge_model] = {
            k: v for k, v in ref_answers[judge_model].items() 
            if k in question_ids
        }

    if mode == "single":
        judges = make_judge_single(judge_model, judge_prompts)
        play_a_match_func = play_a_match_single
        make_match_func = make_match_single
        baseline_model = None
    else:
        judges = make_judge_pairwise(judge_model, judge_prompts)
        play_a_match_func = play_a_match_pair
        if mode == "pairwise-all":
            make_match_func = make_match_all_pairs
            baseline_model = None
        else:
            make_match_func = make_match
            baseline_model = baseline_model

    check_data(questions, model_answers, ref_answers, models, judges)

    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    # Make matches
    matches = []
    matches += make_match_func(
        question_default, models, model_answers, judges["default"], baseline_model
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math"],
        baseline_model,
        ref_answers,
    )
    matches += make_match_func(
        question_default,
        models,
        model_answers,
        judges["default-mt"],
        baseline_model,
        multi_turn=True,
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math-mt"],
        baseline_model,
        ref_answers,
        multi_turn=True,
    )

    match_stat = {}
    match_stat["bench_name"] = "mt_bench"
    match_stat["mode"] = mode
    match_stat["judge"] = judge_model
    match_stat["baseline"] = baseline_model
    match_stat["model_list"] = models
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file

    result_list = []
    # TODO: need to fix the parallel version
    # Play matches
    if num_workers == 1:
        for match in tqdm(matches):
            result = play_a_match_func(match, output_file=None)#output_file=output_file)
            result_str = json.dumps(result) + "\n"
            result_list.append(result_str)
    else:
        assert 0 # not ready yet b/c results list
        def play_a_match_wrapper(match):
            play_a_match_func(match, output_file=output_file)

        np.random.seed(0)
        np.random.shuffle(matches)

        with ThreadPoolExecutor(num_workers) as executor:
            for future in tqdm(executor.map(play_a_match_wrapper, matches), total=len(matches)):
                result_list.append(future)

        result_list = [r for future in result_list for r in future.result()]
                

    with open(output_file, "w") as f:
        f.writelines(result_list)

