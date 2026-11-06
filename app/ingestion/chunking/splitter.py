from dataclasses import dataclass

from tokenizers import Tokenizer

from app.ingestion.parsers.extract import ExtractedDocument

# Tried in order: paragraphs, lines, sentences, words.
SEPARATORS = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    content: str
    token_count: int
    page: int | None


def chunk_document(
    document: ExtractedDocument,
    tokenizer: Tokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> list[TextChunk]:
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    chunks: list[TextChunk] = []
    for page in document.pages:
        for piece in _split(page.text, tokenizer, max_tokens, overlap_tokens, SEPARATORS):
            chunks.append(
                TextChunk(
                    chunk_index=len(chunks),
                    content=piece,
                    token_count=_count(tokenizer, piece),
                    page=page.number,
                )
            )
    return chunks


def _count(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def _join_cost(window: list[str], separator_tokens: int) -> int:
    return separator_tokens if window else 0


def _split(
    text: str,
    tokenizer: Tokenizer,
    max_tokens: int,
    overlap_tokens: int,
    separators: tuple[str, ...],
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if _count(tokenizer, text) <= max_tokens:
        return [text]
    for index, separator in enumerate(separators):
        if separator in text:
            pieces = [piece.strip() for piece in text.split(separator) if piece.strip()]
            finer = separators[index + 1 :]
            return _merge(pieces, separator, tokenizer, max_tokens, overlap_tokens, finer)
    return _hard_split(text, tokenizer, max_tokens, overlap_tokens)


def _merge(
    pieces: list[str],
    separator: str,
    tokenizer: Tokenizer,
    max_tokens: int,
    overlap_tokens: int,
    finer_separators: tuple[str, ...],
) -> list[str]:
    separator_tokens = _count(tokenizer, separator)
    chunks: list[str] = []
    window: list[str] = []
    window_tokens = 0
    fresh = 0  # pieces in the window that were not carried over from the previous chunk

    for piece in pieces:
        piece_tokens = _count(tokenizer, piece)
        if piece_tokens > max_tokens:
            # Too big for any window: flush, then split it with the finer separators.
            if fresh:
                chunks.append(separator.join(window))
            chunks.extend(_split(piece, tokenizer, max_tokens, overlap_tokens, finer_separators))
            window, window_tokens, fresh = [], 0, 0
            continue

        if window_tokens + _join_cost(window, separator_tokens) + piece_tokens > max_tokens:
            if fresh:
                chunks.append(separator.join(window))
                window, window_tokens = _carry_overlap(window, separator_tokens, tokenizer, overlap_tokens)
            fresh = 0
            if window_tokens + _join_cost(window, separator_tokens) + piece_tokens > max_tokens:
                # The carried overlap plus this piece does not fit, so drop the overlap.
                window, window_tokens = [], 0

        window_tokens += _join_cost(window, separator_tokens) + piece_tokens
        window.append(piece)
        fresh += 1

    if fresh:
        chunks.append(separator.join(window))
    return chunks


def _carry_overlap(
    window: list[str],
    separator_tokens: int,
    tokenizer: Tokenizer,
    overlap_tokens: int,
) -> tuple[list[str], int]:
    tail: list[str] = []
    tail_tokens = 0
    for piece in reversed(window):
        cost = _count(tokenizer, piece) + _join_cost(tail, separator_tokens)
        if tail_tokens + cost > overlap_tokens:
            break
        tail.insert(0, piece)
        tail_tokens += cost
    return tail, tail_tokens


def _hard_split(text: str, tokenizer: Tokenizer, max_tokens: int, overlap_tokens: int) -> list[str]:
    # Last resort for text with no separators, such as one very long token run.
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    step = max_tokens - overlap_tokens
    chunks: list[str] = []
    for start in range(0, len(ids), step):
        chunks.append(tokenizer.decode(ids[start : start + max_tokens]))
        if start + max_tokens >= len(ids):
            break
    return chunks
