import torch
import torch.nn as nn

from jabart_llm.config import ModelConfig


class InputEmbeddings(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.dropout = nn.Dropout(config.dropout)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_embedding = nn.Embedding(config.max_seq_len, config.d_model)

    def forward(
        self, input_ids: torch.Tensor, position_offset: int = 0, debug: bool = False
    ):
        _, seq_len = input_ids.shape
        if position_offset + seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {position_offset + seq_len} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )

        positions = torch.arange(
            position_offset, position_offset + seq_len, device=input_ids.device
        )

        tok = self.token_embedding(input_ids)
        pos = self.positional_embedding(positions)

        if debug:
            print(
                f"  input_ids       : {input_ids.tolist()}  shape={tuple(input_ids.shape)}"
            )
            print(f"  position_offset : {position_offset}")
            print(f"  positions       : {positions.tolist()}")
            print(f"  tok shape       : {tuple(tok.shape)}   (B, T, d_model)")
            print(f"  pos shape       : {tuple(pos.shape)}      (T, d_model)")
            same = torch.equal(pos, self.positional_embedding.weight[positions])
            print(f"  pos == table rows {positions.tolist()} ? {same}")
            print(f"  first pos vector (first 4 values): {pos[0, :4].tolist()}")

        encoder_input = self.dropout(tok + pos)

        return encoder_input  #   (B, seq_len, d_model), where B is the number of the sentences, seq_len: the length of sentences,
        #   d_model is the embeddings' dimension
