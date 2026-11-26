import torch
import torch.nn as nn

from jabart_llm.config import ModelConfig
from jabart_llm.model.attention import MultiHeadAttention
from jabart_llm.model.feedforward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layers_norm_eps)
        self.attn = MultiHeadAttention(config=config)

        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layers_norm_eps)
        self.ffn = FeedForward(config=config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        # 1) Attention sub-layer (pre-LN + residual)
        attn_out, updated_cache = self.attn(
            self.ln1(x), kv_cache=kv_cache, use_cache=use_cache
        )
        x = x + attn_out

        # 2) Feed-forward sub-layer (pre-LN + residual)
        x = x + self.ffn(self.ln2(x))

        return x, updated_cache  # x: (B, T, d_model)
