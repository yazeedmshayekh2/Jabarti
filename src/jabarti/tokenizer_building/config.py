"""Configuration for training the bilingual BPE tokenizer."""

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TokenizerConfig:
    """All settings for training the tokenizer."""

    # ── Paths ────────────────────────────────────────────────────────────
    input_files: tuple[Path, ...] = (
        BASE_DIR / "output" / "pretrain_phase1train_cleaned.parquet",
        BASE_DIR / "output" / "pretrain_phase2train_cleaned.parquet",
    )
    tokenizer_path: Path = BASE_DIR / "tokenizer.json"

    # ── Training hyperparameters ─────────────────────────────────────────
    vocab_size: int = 32_000
    min_frequency: int = 2
    batch_size: int = 1_000

    # ── Special tokens (order = token IDs; do not reorder after training) ─
    special_tokens: list[str] = field(default_factory=lambda: [
        "[UNK]", "[PAD]", "[BOS]", "[EOS]", "[SEP]",
        "[AR]", "[EN]",
        "[SYS]", "[ASSISTANT]", "[USER]",
    ])

    @property
    def unk_token(self) -> str:
        return self.special_tokens[0]

CONFIG = TokenizerConfig()