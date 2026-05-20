"""Tests for autocut.pipeline.transcriber — pure helper functions only (no Whisper model)."""

from autocut.pipeline.transcriber import _normalize


# ── _normalize ────────────────────────────────────────────────────────────────

def test_normalize_lowercases():
    """Input is converted to lowercase."""
    assert _normalize("Euh") == "euh"


def test_normalize_strips_trailing_punctuation():
    """Trailing punctuation characters are stripped from the right."""
    assert _normalize("donc,") == "donc"
    assert _normalize("word.") == "word"
    assert _normalize("ah!") == "ah"
    assert _normalize('ok"') == "ok"


def test_normalize_strips_trailing_hyphen():
    """Trailing hyphens (partial-word restarts) are stripped."""
    # Partial-word restart: "donc-" should strip the hyphen
    assert _normalize("donc-") == "donc"


def test_normalize_strips_leading_whitespace():
    """Leading whitespace is removed by strip()."""
    assert _normalize("  euh") == "euh"


def test_normalize_strips_trailing_whitespace():
    """Trailing whitespace is removed by strip()."""
    assert _normalize("euh  ") == "euh"


def test_normalize_empty_string():
    """An empty string normalises to an empty string."""
    assert _normalize("") == ""


def test_normalize_whitespace_only():
    """A whitespace-only string normalises to an empty string."""
    assert _normalize("   ") == ""


def test_normalize_mid_word_hyphen_preserved():
    """Hyphens in the middle of a word are not stripped (rstrip only affects the end)."""
    # Hyphens in the middle of a word are NOT stripped
    assert _normalize("well-known") == "well-known"


def test_normalize_multiple_trailing_punctuation():
    """Multiple consecutive trailing punctuation characters are all stripped."""
    assert _normalize("really?!") == "really"


def test_normalize_nfc_leaves_already_normal_word():
    """A word already in NFC form survives normalisation unchanged."""
    # "voilà" in NFC form should survive normalisation unchanged
    assert _normalize("Voilà") == "voilà"


def test_normalize_strips_semicolons_and_colons():
    """Semicolons and colons at the end of a word are stripped."""
    assert _normalize("ok;") == "ok"
    assert _normalize("ok:") == "ok"
