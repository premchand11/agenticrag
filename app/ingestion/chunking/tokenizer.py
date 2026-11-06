from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer


class TokenizerUnavailable(RuntimeError):
    """The tokenizer file used for chunk sizing is missing."""


@lru_cache
def load_tokenizer(path: str) -> Tokenizer:
    if not Path(path).is_file():
        raise TokenizerUnavailable(f"tokenizer file not found: {path}")
    return Tokenizer.from_file(path)
