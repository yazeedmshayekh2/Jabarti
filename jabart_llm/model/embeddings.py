import torch
import torch.nn as nn

from jabart_llm.tokenizer import Tokenizer
from jabart_llm.config import ModelConfig

class InputEmbeddings(nn.Module):
    def __init__(self, config: ModelConfig):
        self.max_seq_len = config.max_seq_len
        self.dropout = nn.Dropout(config.dropout)

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model
        )

    def forward(self, x: list[int]):
        pass