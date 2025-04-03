from .memmap_loader import MMAPDataset
from .multi_choice_loader import HuggingFaceMultiChoiceDataset
from .posttrain_loader import SFTDataset

try:
    from .streaming_loader import StreamingTextDataset
except ModuleNotFoundError as e:
    print(f"Failed to import StreamingTextDataset: {e}")

from .loader import get_data_loader, InfiniteDataLoader
