from pathlib import Path
from tokenizers import Tokenizer as BackingTokenizer

import json

DEFAULT_TOKENIZER_PATH = Path(__file__).parent / "assets" / "tokenizer.json"

with open(DEFAULT_TOKENIZER_PATH, 'r', encoding="UTF-8") as f:
    result = json.load(f)
    _SPECIAL_TOKENS = {}
    
    for token in result['added_tokens']:
        _SPECIAL_TOKENS[f'{token['content']}']= token['id']

class Tokenizer:
    def __init__(self, backing_tokenizer: BackingTokenizer):
        self._tokenizer = backing_tokenizer

    @classmethod
    def load_tokenizer(cls, json_path=DEFAULT_TOKENIZER_PATH):
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(
                f"No tokenizer at the path: {json_path}."
            )

        return cls(
            BackingTokenizer.from_file( str(json_path) )
        )

    def normalize(self, text: str) -> str:
        return self._tokenizer.normalizer.normalize_str(text)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def tokenize(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).tokens

    def decode(self, ids: list[int], skip_specia_tokens: bool = True) -> str:
        return self._tokenizer.decode(ids=ids, skip_special_tokens=skip_specia_tokens)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

if __name__ == "__main__":
    print("Special Tokens List:\n",_SPECIAL_TOKENS)