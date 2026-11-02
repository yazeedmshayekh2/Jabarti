import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.W_o = nn.Linear(config.d_model, config.d_model)

        self.W_o = nn.Linear(config.max_seq_len, config.max_seq_len)

        self.mask = torch.tril(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool),
        )

        self.register_buffer("mask", self.mask, persistent=False)

    def forward(self, encoder_input: torch.Tensor):

        # atten_input = self.norm(encoder_input)

        batch_size, seq_len, d_model = encoder_input.shape

        q = (
            self.W_q(encoder_input)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        k = (
            self.W_k(encoder_input)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        v = (
            self.W_v(encoder_input)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        scores = q @ k.transponse(-2, -1)
        scores = scores / d_model**0.5
        scores = scores.masked_fill(~self.mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = weights @ v

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return self.W_o(out), weights
