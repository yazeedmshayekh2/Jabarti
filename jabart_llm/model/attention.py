from contourpy import contour_generator
import torch
import torch.nn as nn

from jabart_llm.config import ModelConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.d_k = config.d_k
        self.n_heads = config.n_heads

        # encoder_input ( n * d_dim )
        self.norm = nn.LayerNorm(config.max_seq_len, config.d_model)

        self.W_q = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)

        self.W_o = nn.Linear(config.max_seq_len, config.max_seq_len)

        self.mask = torch.tril(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool),
        )

        self.register_buffer("mask", self.mask, persistent=False)

    def forward(self, encoder_input: torch.Tensor):
        pass
