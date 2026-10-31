"""
Compare the Jabarti BPE tokenizer against well-known tokenizers.

Metrics (per tokenizer, per language):
  - fertility        : tokens per word (lower = more efficient)
  - chars_per_token  : non-space characters per token (higher = more efficient)
  - unk_rate         : share of tokens that are the UNK token
  - roundtrip        : share of texts where decode(encode(x)) == x

Install once:  pip install transformers tiktoken matplotlib
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tokenizers import Tokenizer

from jabarti.tokenizer_building.cleaning import normalize_text
from jabarti.tokenizer_building.config import CONFIG

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

# Text the tokenizer did NOT see during training. If missing, we sample the
# training corpus instead (results will then be biased in favor of Jabarti).
EVAL_FILE = Path("data/eval/held_out.parquet")
SAMPLES_PER_FILE = 2000
SEED = 42

# False: each tokenizer gets the raw text (real-world behavior).
# True : everyone gets normalize_text(text) first (apples-to-apples, since
#        Jabarti's normalizer otherwise strips things like diacritics for free).
NORMALIZE_FOR_ALL = False

# Well-known tokenizers to compare against. Comment out any you don't want.
HF_MODELS = {
    "Qwen2.5": "Qwen/Qwen2.5-0.5B",                  # multilingual byte-level BPE, ~151k vocab
    "AraBERT v2": "aubmindlab/bert-base-arabertv2",  # Arabic-specific WordPiece, ~64k vocab
}
TIKTOKEN_ENCODINGS = {
    "GPT-4o": "o200k_base",                          # OpenAI byte-level BPE, ~200k vocab
}

EXAMPLES = [
    "جُمْهُورِيَّةُ مِصْرَ الْعَرَبِيَّة عاصمتها القاهرة",
    "أحمد إبراهيم آدم وُلد في مصر",
    "The United States and the United Nations signed the agreement!! ! , Machine20",
    "machine learning models require large amounts of training data",
]

OUTPUT_DIR = Path(CONFIG.tokenizer_path).parent

# --------------------------------------------------------------------------- #
# Tokenizer wrappers with one common interface
# --------------------------------------------------------------------------- #


class Candidate:
    name: str
    vocab_size: int
    unk_id: int | None = None

    def ids(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...

    def pieces(self, text: str) -> list[str]:
        """Human-readable pieces (decodes each id on its own)."""
        return [self.decode([i]) for i in self.ids(text)]

    def reference(self, text: str) -> str:
        """What a perfect round-trip should give back."""
        return text


class JabartiCandidate(Candidate):
    def __init__(self, path: Path):
        self.tok = Tokenizer.from_file(str(path))
        self.name = "Jabarti (ours)"
        self.vocab_size = self.tok.get_vocab_size()
        self.unk_id = self.tok.token_to_id(CONFIG.unk_token)

    def ids(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tok.decode(ids)

    def pieces(self, text):
        return self.tok.encode(text, add_special_tokens=False).tokens

    def reference(self, text):
        return normalize_text(text)


class HFCandidate(Candidate):
    def __init__(self, name: str, repo_id: str):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(repo_id)
        self.name = name
        self.vocab_size = len(self.tok)
        self.unk_id = self.tok.unk_token_id

    def ids(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tok.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


class TiktokenCandidate(Candidate):
    def __init__(self, name: str, encoding: str):
        import tiktoken

        self.enc = tiktoken.get_encoding(encoding)
        self.name = name
        self.vocab_size = self.enc.n_vocab
        self.unk_id = None  # byte-level: never produces UNK

    def ids(self, text):
        return self.enc.encode(text, disallowed_special=())

    def decode(self, ids):
        return self.enc.decode(ids)

    def pieces(self, text):
        return [
            self.enc.decode_single_token_bytes(i).decode("utf-8", errors="replace")
            for i in self.ids(text)
        ]


def load_candidates() -> list[Candidate]:
    candidates: list[Candidate] = [JabartiCandidate(CONFIG.tokenizer_path)]
    loaders = [(HFCandidate, n, r) for n, r in HF_MODELS.items()] + [
        (TiktokenCandidate, n, e) for n, e in TIKTOKEN_ENCODINGS.items()
    ]
    for cls, name, source in loaders:
        try:
            candidates.append(cls(name, source))
            print(f"  loaded {name}")
        except Exception as exc:  # missing package, no network, gated repo...
            print(f"  skipped {name}: {exc}")
    return candidates


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")


def detect_lang(text: str) -> str:
    letters = sum(c.isalpha() for c in text)
    arabic = len(ARABIC_RE.findall(text))
    return "Arabic" if letters and arabic / letters > 0.5 else "English"


def load_eval_texts() -> list[str]:
    if EVAL_FILE.exists():
        print(f"  using held-out file: {EVAL_FILE}")
        texts = pd.read_parquet(EVAL_FILE)["text"].tolist()
    else:
        print("  WARNING: no held-out file; sampling the TRAINING corpus "
              "(numbers will flatter Jabarti)")
        texts = []
        for path in CONFIG.input_files:
            df = pd.read_parquet(path)
            texts += df["text"].sample(
                n=min(SAMPLES_PER_FILE, len(df)), random_state=SEED
            ).tolist()
    return [t for t in texts if isinstance(t, str) and t.strip()]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def evaluate(cand: Candidate, texts: list[str]) -> pd.DataFrame:
    rows = []
    for text in texts:
        ids = cand.ids(text)
        decoded = cand.decode(ids)
        rows.append({
            "tokenizer": cand.name,
            "lang": detect_lang(text),
            "n_tokens": len(ids),
            "n_words": len(text.split()),
            "n_chars": len(re.sub(r"\s", "", text)),
            "n_unk": sum(i == cand.unk_id for i in ids) if cand.unk_id is not None else 0,
            "roundtrip": decoded.strip() == cand.reference(text).strip(),
        })
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame, candidates: list[Candidate]) -> pd.DataFrame:
    g = raw.groupby(["tokenizer", "lang"])
    summary = g[["n_tokens", "n_words", "n_chars", "n_unk"]].sum()
    summary["fertility"] = summary.n_tokens / summary.n_words
    summary["chars_per_token"] = summary.n_chars / summary.n_tokens
    summary["unk_rate"] = summary.n_unk / summary.n_tokens
    summary["roundtrip"] = g["roundtrip"].mean()
    summary = summary.reset_index()
    vocab = {c.name: c.vocab_size for c in candidates}
    summary.insert(1, "vocab_size", summary.tokenizer.map(vocab))
    return summary


def plot(summary: pd.DataFrame, path: Path):
    order = summary.tokenizer.unique()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (metric, title, better) in zip(axes, [
        ("fertility", "Tokens per word", "lower is better"),
        ("chars_per_token", "Characters per token", "higher is better"),
        ("roundtrip", "Exact round-trip rate", "higher is better"),
    ]):
        pivot = summary.pivot(index="lang", columns="tokenizer", values=metric)[order]
        pivot.plot.bar(ax=ax, rot=0, edgecolor="black", width=0.8)
        ax.set_title(f"{title}\n({better})")
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=7)
        ax.legend(fontsize=8)
    fig.suptitle("Jabarti vs well-known tokenizers", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  chart saved to: {path}")


def show_examples(candidates: list[Candidate]):
    for text in EXAMPLES:
        text_in = normalize_text(text) if NORMALIZE_FOR_ALL else text
        print("=" * 70)
        print(f"input: {text}")
        for cand in candidates:
            pieces = cand.pieces(text_in)
            print(f"\n  {cand.name:<16} {len(pieces):>3} tokens")
            print(f"  {pieces}")


def main():
    print("STEP 1: LOAD TOKENIZERS")
    candidates = load_candidates()

    print("\nSTEP 2: SIDE-BY-SIDE EXAMPLES")
    show_examples(candidates)

    print("\nSTEP 3: CORPUS-LEVEL METRICS")
    texts = load_eval_texts()
    if NORMALIZE_FOR_ALL:
        texts = [normalize_text(t) for t in texts]
    print(f"  {len(texts)} texts")

    raw = pd.concat([evaluate(c, texts) for c in candidates], ignore_index=True)
    summary = summarize(raw, candidates)

    pd.set_option("display.width", 160)
    print()
    print(summary[["tokenizer", "vocab_size", "lang", "fertility",
                   "chars_per_token", "unk_rate", "roundtrip"]]
          .round(3).to_string(index=False))

    summary.to_csv(OUTPUT_DIR / "tokenizer_comparison.csv", index=False)
    plot(summary, OUTPUT_DIR / "tokenizer_comparison.png")


if __name__ == "__main__":
    main()