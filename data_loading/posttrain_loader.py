import torch
from torch.utils.data import Dataset

from datasets import load_dataset

from utils import Registry, logger
from training.constants import LossConstants


class PosttrainDataset(Dataset):

    def _encode_chat_example(self, example):
        """
        Taken from: https://github.com/allenai/open-instruct/blob/ba11286e5b9eb00d4ce5b40ef4cac1389888416a/open_instruct/finetune.py#L385
        
        This function encodes a single example into a chat format that can be used for post-training.
        Here, we assume each example has a 'messages' field. Each message in it is a dict with 'role' and 'content' fields.
        We use the `apply_chat_template` function from the tokenizer to tokenize the messages and prepare the input and label tensors.
        """
        messages = example["messages"]
        if len(messages) == 0:
            raise ValueError("messages field is empty.")

        kwargs = {
            "tokenize": True,
            "return_tensors": "pt",
            "padding": False,
            "truncation": True,
            "max_length": self.max_length,
        }
        
        input_ids = self.tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=False, **kwargs)
        labels = input_ids.clone()
        
        # mask the non-assistant part for avoiding loss
        for message_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                # we calculate the start index of this non-assistant message
                if message_idx == 0:
                    message_start_idx = 0
                else:
                    message_start_idx = self.tokenizer.apply_chat_template(
                        conversation=messages[:message_idx],
                        add_generation_prompt=False,
                        **kwargs,
                    ).shape[1]
                # next, we calculate the end index of this non-assistant message
                if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                    # for intermediate messages that follow with an assistant message, we need to
                    # set `add_generation_prompt=True` to avoid the assistant generation prefix being included in the loss
                    # (e.g., `<|assistant|>`)
                    message_end_idx = self.tokenizer.apply_chat_template(
                        conversation=messages[:message_idx + 1],
                        add_generation_prompt=True,
                        **kwargs,
                    ).shape[1]
                else:
                    # for the last message or the message that doesn't follow with an assistant message,
                    # we don't need to add the assistant generation prefix
                    message_end_idx = self.tokenizer.apply_chat_template(
                        conversation=messages[:message_idx + 1],
                        add_generation_prompt=False,
                        **kwargs,
                    ).shape[1]
                # set the label to -100 for the non-assistant part
                labels[:, message_start_idx:message_end_idx] = -100
                if self.max_length and message_end_idx >= self.max_length:
                    break
        
        attention_mask = torch.ones_like(input_ids)
        return {
            # remove the last input token b/c there's nothing after to predict 
            "input_ids": input_ids.flatten()[:-1],
            # shift labels back by one because they are next token targets 
            "labels": labels.flatten()[1:],
            # remove 1 from the mask length 
            "attention_mask": attention_mask.flatten()[1:],
        }



@Registry.register("sft_dataset")
class SFTDataset(PosttrainDataset):

    def __init__(self, tokenizer, dataset_path, split, max_length, num_workers, ignore_index=LossConstants.IGNORE_INDEX):
        self.ignore_index = ignore_index
        self.max_length = max_length
        self.num_workers = num_workers

        tokenizer_cls = Registry.get(tokenizer.id)
        self.tokenizer = tokenizer_cls(**tokenizer.args)

        logger.info(f"loading dataset from {dataset_path} with split {split}")
        self.hf_dataset = load_dataset(dataset_path, split=split)

        self.hf_dataset = self.hf_dataset.map(
            self._encode_chat_example,
            batched=False,
            num_proc=self.num_workers,
            remove_columns=[
                name for name in self.hf_dataset.column_names 
                if name not in ["input_ids", "labels", "attention_mask"]
            ],
            desc="Tokenizing and reformatting instruction data",
        )

        # remove examples with no labels
        self.hf_dataset = self.hf_dataset.filter(lambda row: any(x != self.ignore_index for x in row["labels"]))
        self.size = len(self.hf_dataset)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        row = self.hf_dataset[idx]
        return row

