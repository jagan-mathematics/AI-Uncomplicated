from dataclasses import dataclass
from transformers import AutoModelForCausalLM
from torch import nn

from typing import Tuple, List

@dataclass
class ModelArgs:
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"


def get_model_from_hf(model_name):
    return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")


def build_fsdp_grouping_plan(model_args: nn.Module) -> List[Tuple[str, bool]]:
    group_plan: Tuple[int, bool] = []

    # Grouping and output separately
    group_plan.append(("model.embed_tokens", False))

    # Grouping by layers
    for i in range(model_args["num_hidden_layers"]):
        group_plan.append((f"model.layers.{i}", False))

    group_plan.append((f"llm_head", False))
    return group_plan