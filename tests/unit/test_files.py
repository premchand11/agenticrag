import pytest
from app.ingestion.files import UploadRejected, file_type_for, safe_filename, sha256_hex


def test_safe_filename_rejects_a_path():
    with pytest.raises(UploadRejected):
        safe_filename("../secret.html")


def test_file_type_for_accepts_html():
    assert file_type_for("notes.html") == "html"


def test_file_type_for_rejects_txt():
    with pytest.raises(UploadRejected):
        file_type_for("notes.txt")


def test_sha256_hex_is_64_characters():
    assert len(sha256_hex(b"hello")) == 64
