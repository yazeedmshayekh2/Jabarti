import torch
import torch.nn as nn
import torch.nn.functional as F

from jabart_llm.config import ModelConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.d_k = config.d_k
        self.n_heads = config.n_heads

        self.W_q = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)

        self.W_o = nn.Linear(config.d_model, config.d_model)

        self.W_o = nn.Linear(config.max_seq_len, config.max_seq_len)

        self.mask = torch.tril(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool),
        )

        self.register_buffer("mask", self.mask, persistent=False)

    def forward(
        self,
        encoder_input: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        use_cache: bool = False,
    ):
        batch_size, seq_len, d_model = encoder_input.shape

        past_len = 0
        if kv_cache:
            past_k = kv_cache[0]
            past_v = kv_cache[1]

            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        updated_cache = (k, v) if use_cache else None
        total_length = past_len + seq_len

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

        causal = self.mask[past_len:seq_len, :total_length]

        # Causal mask slice (works with and without KV cache)
        #
        # self.mask is a (max_seq_len x max_seq_len) lower-triangular matrix, built once.
        #   past_len     = number of tokens already stored in the KV cache
        #   seq_len      = past_len + number of new tokens (absolute end position)
        #   total_length = all keys = cached + new (same as seq_len here)
        #
        # ROWS (past_len:seq_len) -> the queries = only the NEW tokens.
        #   We start at past_len so each new token keeps its real position.
        #   e.g. past_len=3, 2 new tokens -> rows 3:5
        #
        # COLS (:total_length) -> the keys = ALL tokens (cached + new).
        #   New tokens may attend to everything before them, including the cache.
        #   e.g. cols 0:5
        #
        # Training / no cache: past_len = 0, so this becomes mask[:seq_len, :seq_len]
        #   (the normal square causal mask).
        # Generating 1 token at a time: one row, all ones -> the token sees everything before it.

        scores = scores.masked_fill(
            ~causal, float("-inf")
        )  # ~ will make the zeroes ones and ones zeroes because the mask fill will apply the mask to 1 values not the zeroes.
        # float(-inf) fill the ones with -inf.

        weights = F.softmax(
            scores, dim=-1
        )  # dim=-1 means to apply the softmax at the row level not the column level.

        out = weights @ v

        out = (
            out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )  # .transpose().contiguous().view() this will revert the matrix to its own shape, concatenating the heads outputs.

        return self.W_o(out), updated_cache
