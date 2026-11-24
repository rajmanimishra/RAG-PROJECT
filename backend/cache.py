import hashlib
import math
import re
import unicodedata

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_STOP_WORDS = frozenset(
    "a an and are as at be can could did do does for from give how i in is it "
    "list me of on or please show tell the this to was were what when where "
    "which who why with would you".split()
)
_NEGATIONS = frozenset({"no", "not", "never", "without", "except"})
_NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)*\b")


def normalize_question(question: str) -> str:
    """Normalize equivalent casing, Unicode forms, and whitespace."""
    normalized = unicodedata.normalize("NFKC", question).casefold()
    return " ".join(normalized.split())


def question_cache_field(question: str, document_version: str) -> str:
    """Build a stable exact-match key scoped to one document version."""
    value = f"{document_version}\0{normalize_question(question)}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value)
        for value in (*left, *right)
    ):
        return 0.0

    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot_product / (left_norm * right_norm)


def semantic_match_is_safe(
    question: str,
    candidate_question: str,
    question_embedding: list[float],
    candidate_embedding: list[float],
    threshold: float,
) -> bool:
    """Require a very close embedding and matching question-specific details."""
    question = normalize_question(question)
    candidate_question = normalize_question(candidate_question)
    if not question or not candidate_question:
        return False

    question_terms = {
        token for token in _TOKEN_PATTERN.findall(question)
        if len(token) > 1 and token not in _STOP_WORDS
    }
    candidate_terms = {
        token for token in _TOKEN_PATTERN.findall(candidate_question)
        if len(token) > 1 and token not in _STOP_WORDS
    }
    if not question_terms or not candidate_terms:
        return False

    if set(_NUMBER_PATTERN.findall(question)) != set(
        _NUMBER_PATTERN.findall(candidate_question)
    ):
        return False

    if question_terms & _NEGATIONS != candidate_terms & _NEGATIONS:
        return False

    overlap = len(question_terms & candidate_terms) / max(
        len(question_terms), len(candidate_terms)
    )
    return overlap >= 0.8 and cosine_similarity(
        question_embedding, candidate_embedding
    ) >= threshold
