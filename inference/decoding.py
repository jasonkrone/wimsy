from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from utils import Registry
from model.kv_cache import MultiLayerCache


class DecodingAlgorithm(ABC):

    @abstractmethod
    def sample(self, logits, temperature) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def __call__(
        self,
        prompt_list,
        model,
        stop_criteria,
        temperature,
        pad_token_id,
        max_length,
        device,
    ):
        """
        Returns the output of the model conditioned on the given prompts.

        Prompt: List[List[int]] list of tokenized prompts

        Approach based off: https://github.com/meta-llama/llama3/blob/d3eca219e453eca77ce1e180f48954c7d67467a9/llama/generation.py#L121
        """
        # check that types are correct
        assert isinstance(prompt_list, list) and isinstance(prompt_list[0], list)

        num_prompts = len(prompt_list)
        prompt_lens = [len(p) for p in prompt_list]
        min_prompt_len = min(prompt_lens)
        max_prompt_len = max(prompt_lens)
        assert max_prompt_len < max_length

        out_logits = None
        out_ids = torch.full(size=(num_prompts, max_length), fill_value=pad_token_id, dtype=torch.long, device=device)
        out_lengths = torch.tensor(prompt_lens, dtype=torch.int, device=device)

        # pad prompts
        for i, p in enumerate(prompt_list):
            out_ids[i, :len(p)] = torch.tensor(p, dtype=torch.long)
        mask = out_ids != pad_token_id

        prev_pos = 0
        cur_pos = min_prompt_len
        is_unfinished_seq = torch.ones(num_prompts, dtype=torch.int, device=device)
        has_eos_stop_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stop_criteria)

        # set up cache
        cache = None
        if model.use_cache:
            cache = MultiLayerCache(model.n_layers)

        #pbar.max_length

        while torch.any(is_unfinished_seq):
            input = out_ids[:, prev_pos:cur_pos]
            logits = model(input, cache=cache)
            next_token = self.sample(logits[:, -1:, :], temperature=temperature)
            assert next_token.shape[1] == 1

            # finished seqs should have their next token be a padding token
            # taken from https://github.com/huggingface/transformers/blob/048f599f3506e57e0a595b455d9d2834c8d45023/src/transformers/generation/utils.py#L2519
            if has_eos_stop_criteria:
                next_token = next_token * is_unfinished_seq + pad_token_id * (1 - is_unfinished_seq)

            # only add to output and increment length for prompts that are shorter than cur_pos
            next_token = torch.where(mask[:, cur_pos], out_ids[:, cur_pos], next_token.reshape(-1))
            out_lengths += torch.where(mask[:, cur_pos], 0, is_unfinished_seq)
            out_ids[:, cur_pos] = next_token

            # update logits
            if model.use_cache:
                if out_logits is None:
                    out_logits = logits
                else:
                    out_logits = torch.cat([out_logits, logits], dim=1)
            else:
                out_logits = logits

            if model.use_cache:
                prev_pos = cur_pos

            cur_pos += 1
            is_unfinished_seq = (stop_criteria(out_ids[:, :cur_pos], scores=out_logits) == False).type(torch.int)

        if model.use_cache:
            cache.clear()

        return out_ids, out_logits, out_lengths

# TODO: seems like these should really be samplers e.g., top_p is a logit processor and these are samplers

@Registry.register("greedy_decoding")
class GreedyDecoding(DecodingAlgorithm):

    def sample(self, logits, temperature) -> torch.Tensor:
        """
        logits: (N, T, V)
        """
        return torch.argmax(logits, dim=-1)


@Registry.register("multinomial_decoding")
class MultinomialDecoding(DecodingAlgorithm):

    def sample(self, logits, temperature) -> torch.Tensor:
        """
        logits: (N, T, V)
        """
        _, T, _ = logits.shape
        assert T == 1

        logits = logits.squeeze(1)
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


@Registry.register("top_p_decoding")
class TopPDecoding(DecodingAlgorithm):
    """
    Based on https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
    """

    def sample(self):
        pass
