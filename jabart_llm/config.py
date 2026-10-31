from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 32_000
    d_model: int = 768
    max_seq_len: int = 1024
    dropout: float = 0.1

CONFIG= ModelConfig()
