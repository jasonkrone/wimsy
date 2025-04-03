import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from utils import Registry
from training.constants import LossConstants


@Registry.register("cross_entropy")
class CrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self, reduction="mean", ignore_index=LossConstants.IGNORE_INDEX, **kwargs):
        super().__init__(ignore_index=ignore_index, reduction=reduction, **kwargs)

    def __call__(self, input, target):
        loss = super().__call__(input=input, target=target)
        out_dict = {"loss": loss}
        return out_dict


@Registry.register("cross_entropy_with_z_loss")
class CrossEntropyWithAuxilaryZLoss(nn.CrossEntropyLoss):

    def __init__(self, z_loss_coefficient, reduction="mean", ignore_index=LossConstants.IGNORE_INDEX, **kwargs):
        super().__init__(ignore_index=ignore_index, reduction=reduction, **kwargs)
        self.z_loss_coefficient = z_loss_coefficient

    def __call__(self, input, target):
        ce_loss = super().__call__(input=input, target=target)
        mask = target != self.ignore_index
        z_loss = input.logsumexp(-1).pow(2)
        z_loss = (z_loss * mask).sum() / mask.sum()
        loss = ce_loss + self.z_loss_coefficient * z_loss
        out_dict = {"loss": loss}
        return out_dict


@Registry.register("length_normalized_dpo_loss")
class DPOLoss(_Loss):

    def __init__(self, beta, ignore_index=LossConstants.IGNORE_INDEX):
        super().__init__()
        self.beta = beta
        self.ignore_index = ignore_index

    def __call__(
        self, 
        policy_logits_accept,
        policy_logits_reject,
        reference_logits_accept,
        reference_logits_reject,
        targets_accept,
        targets_reject,
    ):
        """
        Logits: (batch-size, seq-length, vocab-size)
        Targets: (batch-size, seq-length)

        Implementation based on OpenInstruct/dpo_utils.py dpo_loss function
        """
        policy_logprobs_accept = self.length_normalized_log_softmax(policy_logits_accept, targets_accept)
        policy_logprobs_reject = self.length_normalized_log_softmax(policy_logits_reject, targets_reject)

        reference_logprobs_accept = self.length_normalized_log_softmax(reference_logits_accept, targets_accept)
        reference_logprobs_reject = self.length_normalized_log_softmax(reference_logits_reject, targets_reject)

        policy_ratio = policy_logprobs_accept - policy_logprobs_reject 
        reference_ratio = reference_logprobs_accept - reference_logprobs_reject 
        delta = policy_ratio - reference_ratio
        loss = -F.logsigmoid(delta * self.beta)

        reward_accept = self.beta * (policy_logprobs_accept - reference_logprobs_accept).detach()
        reward_reject = self.beta * (policy_logprobs_reject - reference_logprobs_reject).detach()

        out_dict = {"loss": loss.mean(), "reward_accept": reward_accept, "reward_reject": reward_reject}
        return out_dict

    def length_normalized_log_softmax(self, logits, target):
        mask = target != self.ignore_index
        # to ensure the gather doesn't break - this is masked so no effect 
        target[target == self.ignore_index] = 0
        # normalize probs
        log_probs = logits.log_softmax(dim=-1)
        # get probs of targets
        log_probs = torch.gather(log_probs, dim=2, index=target.unsqueeze(-1)).squeeze()
        log_probs = (log_probs * mask).sum(dim=-1) / mask.sum(dim=-1)
        return log_probs
