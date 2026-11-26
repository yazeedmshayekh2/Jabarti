"""
config.py --  the single source of truth for every model dimension
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32_000
    d_model: int = 768  # Embedding Dim width
    d_ff: int = 4 * 768
    max_seq_len: int = 1024
    dropout: float = 0.1

    n_heads: int = 12
    n_layers: int = 12  # How many TransformerBlocks to stack
    qkv_bias: bool = False
    layers_norm_eps: float = 1e-5  # Used by nn.LayerNorm in the block

    def __post_init__(self):  # Fail early if heads don't divide evenly
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})",
        )

    @property
    def d_k(self):
        return self.d_model // self.n_heads


CONFIG = ModelConfig()
