from torch import dtype, device
from typing import Any
import LLM
def load_model(model_name: str, device: device, dtype: dtype, cache_dir: str = None, model_args = None) -> Any:
    model = LLM.HuggingFaceModel(model_name, device, dtype, cache_dir, **model_args)
    return model
