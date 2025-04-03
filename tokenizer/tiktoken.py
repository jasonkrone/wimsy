import pkg_resources

import torch
import tiktoken

from utils import Registry


@Registry.register("tiktoken")
class TiktokenWrapper(object):
    """
    Wrapper around tiktokenizer to provide the same interface as Huggingface tokenizers.
    Adds the special tokens used for llama3 tokenizer and zephyr chat template.
    """

    REQUIRED_VERSION = "0.6.0"
    # pad added not in llama
    PAD_TOKEN_STR = "<|pad|>"
    BOS_TOKEN_STR = "<|beginoftext|>"
    EOS_TOKEN_STR = "<|endoftext|>"
    # end of turn
    EOT_TOKEN_STR = "<|eot_id|>"
    BOH_TOKEN_STR = "<|start_header_id|>"
    EOH_TOKEN_STR = "<|end_header_id|>"

    USER_TOKEN_STR = "<|user|>"
    AGENT_TOKEN_STR = "<|assistant|>"
    SYSTEM_TOKEN_STR = "<|system|>"

    USER_ROLE_STR = "user"
    AGENT_ROLE_STR = "assistant"
    SYSTEM_ROLE_STR = "system"

    ROLE_NAME_TO_TOKEN = {
        USER_ROLE_STR: USER_TOKEN_STR,
        AGENT_ROLE_STR: AGENT_TOKEN_STR,
        SYSTEM_ROLE_STR: SYSTEM_TOKEN_STR,
    }

    ALLOWED_SPECIAL_TOKENS = [
        PAD_TOKEN_STR,
        BOS_TOKEN_STR,
        EOS_TOKEN_STR,
        EOT_TOKEN_STR,
        BOH_TOKEN_STR,
        EOH_TOKEN_STR,
        USER_TOKEN_STR,
        AGENT_TOKEN_STR,
        SYSTEM_TOKEN_STR,
    ]

    EXTRA_SPECIAL_TOKENS = ALLOWED_SPECIAL_TOKENS + [
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
    ]

    def __init__(self, name="cl100k_add_special"):
        # check tiktoken version to ensure that it's the same as used for pre-train tokenization
        assert pkg_resources.get_distribution("tiktoken").version == self.REQUIRED_VERSION

        self.name = name
        cl100k_base = tiktoken.get_encoding("cl100k_base")
        special_tokens = {
            token: cl100k_base.n_vocab + i for i, token in enumerate(self.EXTRA_SPECIAL_TOKENS)
        }
        special_tokens.update(cl100k_base._special_tokens)

        self.encoding = tiktoken.Encoding(
            name=self.name,
            pat_str=cl100k_base._pat_str,
            mergeable_ranks=cl100k_base._mergeable_ranks,
            special_tokens=special_tokens,
        )

        self.allowed_special_tokens = set(self.ALLOWED_SPECIAL_TOKENS)
        self.pad_token_id = special_tokens[self.PAD_TOKEN_STR]

    @property
    def special_tokens_map(self):
        return {k: k for k in self.encoding._special_tokens}

    @property
    def eos_token(self):
        return self.EOS_TOKEN_STR

    @property
    def pad_token(self):
        return self.PAD_TOKEN_STR

    def __call__(self, text, **kwargs):
        return self.encode(text, return_dict=True, **kwargs)

    def encode(self, text, padding=False, max_length=None, truncation=False, return_tensors=None, return_dict=False, **kwargs):
        assert not kwargs.pop("add_special_tokens", False)
        if isinstance(text, str):
            out = self.encoding.encode(text, allowed_special=self.allowed_special_tokens)
        elif isinstance(text, list):
            out = self.encoding.encode_batch(text, allowed_special=self.allowed_special_tokens)
        else:
            message = f"type: {type(text)} not supported by tiktoken encode. Type must be str or list."
            raise ValueError(message)

        if len(out) and (truncation or padding):
            assert max_length is not None
            out = self._trucate_and_pad(out, max_length, padding, truncation)

        if return_tensors == "pt":
            out = torch.tensor(out)
            if out.dim() == 1:
                out = out.unsqueeze(0)

        if return_dict:
            out = {"input_ids": out}
        return out

    def _trucate_and_pad(self, token_ids, max_length, padding, truncation):
        is_batched = isinstance(token_ids[0], list)
        if not is_batched:
            token_ids = [token_ids]

        for i, ids in enumerate(token_ids):
            if truncation:
                ids = ids[:max_length]
            if padding:
                assert len(ids) <= max_length
                ids += [self.pad_token_id]*(max_length - len(ids))
            token_ids[i] = ids

        if not is_batched:
            token_ids = token_ids[0]
        return token_ids

    def decode(self, token_ids):
        return self.encoding.decode(token_ids)

    def batch_decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids[0], int):
            token_ids = map(lambda x: [x], token_ids)
        return self.encoding.decode_batch(token_ids)

    def apply_chat_template(
        self, 
        conversation,
        add_generation_prompt = False,
        tokenize = True,
        padding = False,
        truncation = False,
        max_length = None,
        return_tensors = None,
        return_dict = False,
        return_assistant_tokens_mask = False,
        tokenizer_kwargs = {},
        **kwargs,
    ):
        out = self._render_template(conversation, add_generation_prompt)
        if tokenize:
            out = self.encode(
                out, padding=padding, truncation=truncation, max_length=max_length, 
                return_tensors=return_tensors, return_dict=return_dict, **tokenizer_kwargs
            )
        if return_assistant_tokens_mask:
            raise ValueError("return_assistant_tokens_mask not yet supported")
        return out

    def _render_template(self, conversation, add_generation_prompt):
        rendered = []
        is_batched = isinstance(conversation[0], list)

        if not is_batched:
            conversation = [conversation]

        for message_list in conversation:
            turns = []
            for i, message in enumerate(message_list):
                special_token = self.ROLE_NAME_TO_TOKEN[message["role"]] 
                turns.append(f"{special_token}\n{message['content']}{self.EOS_TOKEN_STR}")
                if i == len(message_list) - 1 and add_generation_prompt:
                    turns.append(f"{self.AGENT_TOKEN_STR}")

            rendered.append("\n".join(turns) + "\n")

        if not is_batched:
            rendered = rendered[0]
        return rendered
