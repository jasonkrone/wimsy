import os
import sys
import pdb

from tqdm import tqdm
import pandas as pd
from huggingface_hub import login

sys.path.append("./lm-evaluation-harness")
import lm_eval.tasks as eval_tasks


class EvalHarnessStats(object):

    def __init__(self, input_datasets, include_path):
        self.include_path = include_path
        self.input_datasets = input_datasets

    def __call__(self):
        df_dict = {"task": []}
        split_list = ["train", "dev", "test"]
        column_order = ["task"]
        for split in split_list:
            for key in ["n_total", "n_with_text", "n_none"]:
                df_dict[f"{split}_{key}"] = []
                column_order.append(f"{split}_{key}")

        task_manager = eval_tasks.TaskManager("INFO", include_path=os.path.expanduser(self.include_path))
        task_names = task_manager.match_tasks(self.input_datasets)
        task_dict = eval_tasks.get_task_dict(task_names, task_manager)

        for task_name, task in tqdm(task_dict.items(), desc="writing eval harness docs"):
            if isinstance(task, tuple):
                _, task = task
                if task is None:
                    continue

            print(f"========{task_name}===========")

            df_dict["task"].append(task_name)
            for split in split_list:
                counts_dict = self.count_docs(task, split)
                for k, v in counts_dict.items():
                    df_dict[f"{split}_{k}"].append(v)

        df = pd.DataFrame(df_dict)
        df = df[column_order]
        return df

    def count_docs(self, task, split):
        n_none = 0
        n_with_text = 0
        n_total = 0

        if self.task_has_split(task, split):
            docs = self.get_split(task, split)
            for i, doc in enumerate(docs):
                n_total += 1
                text = task.doc_to_text(doc)
                if text is None:
                    n_none += 1
                else:
                    n_with_text += 1
        return {"n_total": n_total, "n_with_text": n_with_text, "n_none": n_none}

    @staticmethod
    def task_has_split(task, split):
        if split == "train":
            has_split = task.has_training_docs()
        elif split == "dev":
            has_split = task.has_validation_docs()
        elif split == "test":
            has_split = task.has_test_docs()
        else:
            raise ValueError(f"split {split} not supported")
        return has_split

    @staticmethod
    def get_split(task, split):
        if split == "train":
            docs = task.training_docs()
        elif split == "dev":
            docs = task.validation_docs()
        elif split == "test":
            docs = task.test_docs()
        else:
            raise ValueError(f"split {split} not supported")
        return docs


def main():
    login(token=os.environ["HUGGINGFACE_TOKEN"])

    dataset_list = [
        "arc_easy",
        "arc_challenge",
        "bbh_cot_fewshot",
        "boolq",
        "commonsense_qa_cot_fewshot",
        "drop",
        "gpqa_main_cot_zeroshot",
        "gsm8k_cot",
        "hellaswag",
        "humaneval_greedy",
        "ifeval",
        "lambada_openai",
        "minerva_math",
        "mmlu",
        "openbookqa",
        "piqa",
        "race_llama",
        "triviaqa_wiki",
        "winogrande",
        "truthfulqa_mc2",
    ]
    task_path = "./tasks"
    save_path = "./figures/eval_harness_stats.csv"
    stats = EvalHarnessStats(dataset_list, task_path)
    stats_df = stats()
    stats_df.to_csv(save_path)


if __name__ == "__main__":
    main()

