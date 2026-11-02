import math

import torch
import torch.nn as nn

try:
    from jabart_llm.config import ModelConfig
except ImportError:  # lets this file run standalone for the demo below
    from dataclasses import dataclass

    @dataclass
    class ModelConfig:
        vocab_size: int = 20
        d_model: int = 8
        max_seq_len: int = 6
        dropout: float = 0.1


torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)


class InputEmbeddings(nn.Module):
    """Token embeddings (learned) + sinusoidal positional encoding (fixed, not learned)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.dropout = nn.Dropout(config.dropout)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # A buffer moves with .to(device) but is NOT a parameter: the optimizer never touches it.
        self.register_buffer(
            "pe",
            self._build_sinusoidal_table(config.max_seq_len, config.d_model),
            persistent=False,
        )

    @staticmethod
    def _build_sinusoidal_table(
        max_seq_len: int, d_model: int, debug: bool = False
    ) -> torch.Tensor:
        # (L, 1): one row per position -> [[0], [1], [2], ...]
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)

        # freq_i = 1 / 10000^(2i / d_model), computed in log space for stability.
        # One frequency per PAIR of dimensions (sin, cos), so d_model/2 values.
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        # (L, 1) * (d_model/2,) -> (L, d_model/2): angle for every (position, pair)
        angles = position * div_term

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(angles)  # even dims -> sin
        pe[:, 1::2] = torch.cos(angles[:, : d_model // 2])  # odd dims -> cos

        if debug:
            print("=" * 70)
            print("BUILDING THE SINUSOIDAL TABLE (happens once, no training involved)")
            print("=" * 70)
            print(f"max_seq_len={max_seq_len}, d_model={d_model}\n")
            print("[1] position column, shape", tuple(position.shape))
            print(position.flatten().tolist(), "\n")
            print(
                f"[2] frequencies 1/10000^(2i/d_model), one per dim PAIR, shape {tuple(div_term.shape)}"
            )
            for i, f in enumerate(div_term.tolist()):
                print(
                    f"    pair {i} (dims {2*i},{2*i+1}): freq = {f:.5f}"
                    f"  -> wavelength ~ {2 * math.pi / f:.1f} positions"
                )
            print("    (first pair is fast, last pair is very slow)\n")
            print(
                f"[3] angles = position * freq, shape {tuple(angles.shape)}  (rows=positions, cols=pairs)"
            )
            print(angles, "\n")
            print(
                "[4] pe[:, even dims] = sin(angles)  |  pe[:, odd dims] = cos(angles)"
            )
            print(
                f"    final table shape {tuple(pe.shape)}  (rows=positions, cols=dims)"
            )
            print(
                "    dim:      "
                + "  ".join(
                    f"{('sin' if d % 2 == 0 else 'cos') + str(d):>6}"
                    for d in range(d_model)
                )
            )
            for p in range(max_seq_len):
                print(
                    f"    pos {p}:   " + "  ".join(f"{v:6.3f}" for v in pe[p].tolist())
                )
            print()
        return pe  # (max_seq_len, d_model)

    def forward(
        self, input_ids: torch.Tensor, position_offset: int = 0, debug: bool = False
    ):
        _, seq_len = input_ids.shape
        if position_offset + seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {position_offset + seq_len} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )

        tok = self.token_embedding(input_ids)  # (B, T, d_model)  learned
        pos = self.pe[
            position_offset : position_offset + seq_len
        ]  # (T, d_model)  fixed
        summed = tok + pos  # pos broadcasts over the batch dimension B
        encoder_input = self.dropout(summed)

        if debug:
            print("=" * 70)
            print("FORWARD PASS")
            print("=" * 70)
            print(
                f"[1] input_ids       : {input_ids.tolist()}  shape={tuple(input_ids.shape)}  (B, T)"
            )
            print(f"[2] position_offset : {position_offset}")
            print(
                f"    rows read from pe: {list(range(position_offset, position_offset + seq_len))}"
            )
            print(
                f"[3] tok = token_embedding(input_ids)   shape={tuple(tok.shape)}  (B, T, d_model)"
            )
            print("    LEARNED: random at start, changes during training. Sentence 0:")
            print(tok[0])
            print(
                f"[4] pos = pe[{position_offset}:{position_offset + seq_len}]   shape={tuple(pos.shape)}  (T, d_model)"
            )
            print("    FIXED: identical every run, never trained:")
            print(pos)
            print(
                "[5] tok + pos (sentence 0): same token at different positions now differs"
            )
            print(summed[0])
            if input_ids.shape[0] > 1:
                same = torch.equal(
                    pos, self.pe[position_offset : position_offset + seq_len]
                )
                print(
                    f"    the same pos matrix is added to all {input_ids.shape[0]} sentences: {same}"
                )
            print(f"[6] dropout (p={self.dropout.p}, training={self.training}):")
            if self.training:
                zeroed = int((encoder_input == 0).sum())
                print(
                    f"    {zeroed} of {encoder_input.numel()} values zeroed, rest scaled by 1/(1-p)"
                )
            else:
                print(
                    "    eval mode -> dropout does nothing, output == tok + pos:",
                    torch.equal(encoder_input, summed),
                )
            print("[7] trainable parameters:", [n for n, _ in self.named_parameters()])
            print("    ('pe' is absent: it is a buffer, not a parameter)")
            print(f"    output shape = {tuple(encoder_input.shape)}  (B, T, d_model)\n")

        return encoder_input  # (B, seq_len, d_model)


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=20, d_model=8, max_seq_len=6, dropout=0.1)

    # Part 1: how the fixed table is built
    InputEmbeddings._build_sinusoidal_table(cfg.max_seq_len, cfg.d_model, debug=True)

    # Part 2: how it is used. The SAME token id (7) appears at positions 0 and 2.
    model = InputEmbeddings(cfg).eval()
    ids = torch.tensor([[7, 3, 7]])
    out = model(ids, debug=True)

    print("=" * 70)
    print("PROOF: token 7 appears twice, but the outputs differ because of position")
    print("=" * 70)
    print(
        "tok(7) identical both times :",
        torch.equal(model.token_embedding(ids[0, 0]), model.token_embedding(ids[0, 2])),
    )
    print("output at pos 0 == output at pos 2 ?", torch.equal(out[0, 0], out[0, 2]))

    # Part 3: offset (e.g. generating one token at a time with a cache)
    print("\nOFFSET: feed one new token that sits at position 3")
    model(torch.tensor([[5]]), position_offset=3, debug=True)
