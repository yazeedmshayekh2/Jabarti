import torch
import torch.nn as nn
import torch.nn.functional as F

from jabart_llm.config import ModelConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.d_k = config.d_k
        self.n_heads = config.n_heads

        # self.norm = nn.LayerNorm(config.max_seq_len, config.d_model)

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

        """
        .view(B, T, n_heads, d_k).transpose(1, 2) this is used for the multi-headed attention,
        the only thing is the previous d_model will be splitted into n_heads, d_k like from (1, 4, 8) -> (1, 4, 2, 4) with transpose
        it will be (1, 2, 4, 4) because the matmul multiplies the last two dims.
        """

        scores = q @ k.transponse(-2, -1)
        scores = scores / d_model**0.5
        scores = scores.masked_fill(
            ~self.mask, float("-inf")
        )  # ~ will make the zeroes ones and ones zeroes because the mask fill will apply the mask to 1 values not the zeroes.
        # float(-inf) fill the ones with -inf.

        weights = F.softmax(
            scores, dim=-1
        )  # dim=-1 means to apply the softmax at the row level not the column level.

        out = weights @ v

        out = (
            out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )  # .transpose().contiguous().view() this will revert the matrix to its own shape, concatenating the heads outputs.

        return self.W_o(out), weights
