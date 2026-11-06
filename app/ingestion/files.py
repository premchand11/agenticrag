import hashlib
from pathlib import Path

_TYPES = {".pdf": "pdf", ".docx": "docx", ".html": "html"}


class UploadRejected(Exception):
    def __init__(self, detail: str) -> None:
        self.detail = detail


class UploadTooLarge(Exception):
    pass


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    if not name or name in {".", ".."} or "/" in filename or "\\" in filename:
        raise UploadRejected("invalid filename")
    return name


def file_type_for(filename: str) -> str:
    file_type = _TYPES.get(Path(filename).suffix.lower())
    if file_type is None:
        raise UploadRejected("file type must be pdf, docx, or html")
    return file_type


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
