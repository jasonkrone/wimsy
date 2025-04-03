from utils import Registry
import transformers

from .tiktoken import TiktokenWrapper

# register huggingface auto tokenizer
Registry.register("hf_auto_tokenizer", "from_pretrained")(transformers.AutoTokenizer)
