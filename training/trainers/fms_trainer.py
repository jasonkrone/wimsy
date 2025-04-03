from training.trainers.trainer import Trainer
from utils import Registry


@Registry.register("hf_trainer")
class FMSTrainer(Trainer):

    @classmethod
    def get_logits(cls, model, input_dict):
        logits = model(input_dict["input_ids"], attn_algorithm="flash")
        return logits
 
