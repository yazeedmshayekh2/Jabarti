"""
Build the BPE Tokenizer (Bilingual)
"""

import token
from typing import Iterator

import pandas as pd
from jabarti.tokenizer_building.cleaning import normalize_text
from jabarti.tokenizer_building.config import TokenizerConfig, CONFIG

from jabarti.tokenizer_building.cleaning import build_normalizer
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

class ScratchTokenizer():
    """Builds and trains a BPE tokenizer from parquet text corpora.

    Attributes:
        vocab_size: Target vocabulary size, including special tokens.
        min_frequency: Minimum pair frequency required for a merge.
        special_tokens: Tokens reserved at the start of the vocabulary, in ID order.
        unk_token: Token used for unknown input; must be in ``special_tokens``.
        text_column: Name of the column holding raw text in the parquet files.
        batch_size: Number of rows read per batch while streaming.
    """

    def __init__(self, tokenizer_config: TokenizerConfig):
        self.config = tokenizer_config

    def build_tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer(model=models.BPE(unk_token=self.config.unk_token))
        tokenizer.normalizer    = build_normalizer()
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme="always")
        tokenizer.decoder       = decoders.Metaspace(prepend_scheme="always")
        return tokenizer

    def corpus_generator(self, df: pd.DataFrame) -> Iterator[list[str]]:
        texts = df["text"].tolist()
        for i in range(0, len(texts), self.config.batch_size):
            yield texts[i: i + self.config.batch_size]

    def train_tokenizer(self, df: pd.DataFrame) -> Tokenizer:
        tokenizer = self.build_tokenizer()
        trainer = trainers.BpeTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            special_tokens=self.config.special_tokens,
            show_progress=True,
        )

        tokenizer.train_from_iterator(
            iterator=self.corpus_generator(df), 
            trainer=trainer,
            length=(len(df) + self.config.batch_size - 1) // self.config.batch_size,
        )

        self.save_tokenizer(tokenizer=tokenizer)

        return tokenizer

    def save_tokenizer(self, tokenizer: Tokenizer):       
        tokenizer.save(str(self.config.tokenizer_path))
        print(f"The tokenizer.json was save to: {self.config.tokenizer_path}")

def main():

    print("STEP 1: LOAD CLEANED CORPUS")
    frames=[]
    for clean_path in CONFIG.input_files:
        frame = pd.read_parquet(clean_path)
        print(f"{clean_path.name}: {len(frame)}")
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)

    print("STEP 2: BUILD & TRAIN THE TOKENIZER")

    scratch_tokernizer = ScratchTokenizer(tokenizer_config=CONFIG)
    trained_tokenizer = scratch_tokernizer.train_tokenizer(df=df)

    print("SANITY CHECK: encode/ decode bilingual examples")
    examples = [
        "جُمْهُورِيَّةُ مِصْرَ الْعَرَبِيَّة عاصمتها القاهرة",
        "أحمد إبراهيم آدم وُلد في مصر",
        "The United States and the United Nations signed the agreement!! ! , Machine20",
        "machine learning models require large amounts of training data",
    ]

    tokens = 0
    words = 0

    print("="*30)

    for text in examples:
        encoded = trained_tokenizer.encode(text, add_special_tokens=False)
        decoded_text = trained_tokenizer.decode(encoded.ids)

        tokens += len(encoded.tokens)
        words += len(text.split())

        print(f"\n  input   : {text}")

        print(f"    tokens  : {encoded.tokens}")
        print(f"    ids     : {encoded.ids}")

        print(f"    decoded : {decoded_text}")

        print(f" Is Equal: {decoded_text == normalize_text(text)}")

        print("="*30)

    print(f"Average tokens per word: {tokens / words:.2}")

if __name__ == "__main__":
    main()