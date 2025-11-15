from dataclasses import dataclass


@dataclass
class HRMCoreConfig:
    vocab_size: int
    hidden_size: int = 768
    num_heads: int = 12
    ff_mult: int = 4
    max_position_embeddings: int = 512

    # Hierarchical controller
    H_layers: int = 3
    L_layers: int = 3
    H_cycles: int = 4   # tăng mặc định để soft‑halting có đất dụng
    L_cycles: int = 4

    # I/O
    tie_embeddings: bool = False