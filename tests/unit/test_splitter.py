from pathlib import Path
from types import SimpleNamespace

import pytest
from app.ingestion.chunking.splitter import _hard_split, chunk_document
from app.ingestion.parsers.extract import ExtractedDocument, ExtractedPage
from tokenizers import Tokenizer, models, pre_tokenizers

BGE_TOKENIZER = Path("models/bge-small-en-v1.5/tokenizer.json")


def _word_tokenizer(*texts: str) -> Tokenizer:
    words = sorted({word for text in texts for word in text.split()})
    vocab = {"[UNK]": 0, **{word: index for index, word in enumerate(words, start=1)}}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tokenizer


def _document(*pages: str) -> ExtractedDocument:
    numbered = tuple(ExtractedPage(number=index, text=text) for index, text in enumerate(pages, start=1))
    return ExtractedDocument(text="\n\n".join(pages), pages=numbered)


class _CharTokenizer:
    """One token per character, so window sizes are exact."""

    def encode(self, text: str, add_special_tokens: bool = False):
        return SimpleNamespace(ids=[ord(char) for char in text])

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(value) for value in ids)


def test_short_text_is_one_chunk():
    text = "Redis caches data"
    chunks = chunk_document(_document(text), _word_tokenizer(text), max_tokens=10, overlap_tokens=2)
    assert len(chunks) == 1
    assert chunks[0].content == text
    assert chunks[0].token_count == 3
    assert chunks[0].page == 1


def test_long_text_is_split_within_the_token_limit():
    text = " ".join(f"w{i}" for i in range(30))
    chunks = chunk_document(_document(text), _word_tokenizer(text), max_tokens=10, overlap_tokens=3)
    assert len(chunks) > 1
    assert all(chunk.token_count <= 10 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_share_the_overlap():
    text = " ".join(f"w{i}" for i in range(30))
    chunks = chunk_document(_document(text), _word_tokenizer(text), max_tokens=10, overlap_tokens=3)
    for previous, following in zip(chunks, chunks[1:], strict=False):
        assert following.content.split()[:3] == previous.content.split()[-3:]


def test_no_words_are_lost_between_chunks():
    text = " ".join(f"w{i}" for i in range(30))
    chunks = chunk_document(_document(text), _word_tokenizer(text), max_tokens=10, overlap_tokens=3)
    seen = {word for chunk in chunks for word in chunk.content.split()}
    assert seen == set(text.split())


def test_chunks_never_cross_page_boundaries():
    first, second = "alpha beta gamma delta", "omega sigma"
    chunks = chunk_document(
        _document(first, second),
        _word_tokenizer(first, second),
        max_tokens=10,
        overlap_tokens=2,
    )
    assert [chunk.content for chunk in chunks] == [first, second]
    assert [chunk.page for chunk in chunks] == [1, 2]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]


def test_blank_page_produces_no_chunks():
    chunks = chunk_document(_document("  \n  "), _word_tokenizer("x"), max_tokens=10, overlap_tokens=2)
    assert chunks == []


def test_overlap_must_be_smaller_than_the_chunk_size():
    text = "a b c"
    with pytest.raises(ValueError):
        chunk_document(_document(text), _word_tokenizer(text), max_tokens=5, overlap_tokens=5)


def test_hard_split_windows_text_without_separators():
    pieces = _hard_split("abcdefghijklmnopqrstuvwxyz", _CharTokenizer(), max_tokens=10, overlap_tokens=2)
    assert pieces == ["abcdefghij", "ijklmnopqr", "qrstuvwxyz"]


def test_real_bge_tokenizer_respects_the_limit():
    if not BGE_TOKENIZER.is_file():
        pytest.skip("download models/bge-small-en-v1.5/tokenizer.json first")
    tokenizer = Tokenizer.from_file(str(BGE_TOKENIZER))
    text = "Redis is an in-memory data store. " * 40
    chunks = chunk_document(_document(text), tokenizer, max_tokens=64, overlap_tokens=8)
    assert len(chunks) > 1
    assert all(chunk.token_count <= 64 for chunk in chunks)
