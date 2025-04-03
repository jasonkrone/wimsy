"""
Makes building block layers initializable by overriding the reset_parameters
method to call the given initializer. Makes it easy to use custom init with
FSDP.
"""
import torch.nn as nn

from utils import Registry

DEFAULT_INITIALIZER = "default_init"


@Registry.register("linear_custom_init")
class LinearCustomInit(nn.Linear):

    def __init__(self, *args, initializer=DEFAULT_INITIALIZER, **kwargs):
        self.initializer = Registry.get(initializer)
        self.init_kwargs = self.initializer.get_kwargs()
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        if self.initializer:
            self.initializer._init_params(self, **self.init_kwargs)
        else:
            super().reset_parameters()


@Registry.register("embed_custom_init")
class EmbeddingCustomInit(nn.Embedding):

    def __init__(self, *args, initializer=DEFAULT_INITIALIZER, **kwargs):
        self.initializer = Registry.get(initializer)
        self.init_kwargs = self.initializer.get_kwargs()
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        if self.initializer:
            self.initializer._init_params(self, **self.init_kwargs)
            self._fill_padding_idx_with_zero()
        else:
            super().reset_parameters()

