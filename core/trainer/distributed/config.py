from dataclasses import dataclass
from typing import Optional


@dataclass
class DistributedArgs:
    dp_shard: int = (
        1
    )
    dp_replicate: int = (
        1  # How many times to replicate the model weight. Typically number of nodes.
    )
    tp_size: int = 1
    selective_activation_checkpointing: bool = False
    compile: bool = False
    fsdp_type: str = "no_shard"
    model_dtype: str = "bf16"
    float8_recipe: Optional[str] = None
    float8_filter: str = r"layers\.[0-9]+\."

    matmul_allow_tf32: bool = False
    allow_bf16_reduced_precision_reduction = True
    detect_anomaly: bool = False

    compile_cache_size_limit: int = 8

    spawn_method: str = "forkserver"