import argparse
from utils import Config, Registry
from training import Trainer, HFTrainer


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to training config file in yaml format")


def train_model(config):
    trainer_cls = Registry.get(config.trainer.id)
    trainer = trainer_cls()
    best_ckpt, ckpt_to_dev_loss = trainer.train(config)
    return best_ckpt, ckpt_to_dev_loss


if __name__ == "__main__":
    args = parser.parse_args()
    train_config = Config.from_yaml(args.config)
    train_model(train_config)

