import pytest
from app.ingestion.chunking.tokenizer import TokenizerUnavailable, load_tokenizer


def test_missing_tokenizer_file_raises_a_clear_error(tmp_path):
    with pytest.raises(TokenizerUnavailable):
        load_tokenizer(str(tmp_path / "missing.json"))
