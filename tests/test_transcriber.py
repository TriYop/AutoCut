"""Tests for autocut.pipeline.transcriber — pure helper functions only (no Whisper model)."""

from autocut.pipeline.transcriber import _normalize


# ── _normalize ────────────────────────────────────────────────────────────────

def test_normalize_lowercases():
    assert _normalize("Euh") == "euh"


def test_normalize_strips_trailing_punctuation():
    assert _normalize("donc,") == "donc"
    assert _normalize("word.") == "word"
    assert _normalize("ah!") == "ah"
    assert _normalize('ok"') == "ok"


def test_normalize_strips_trailing_hyphen():
    # Partial-word restart: "donc-" should strip the hyphen
    assert _normalize("donc-") == "donc"


def test_normalize_strips_leading_whitespace():
    assert _normalize("  euh") == "euh"


def test_normalize_strips_trailing_whitespace():
    assert _normalize("euh  ") == "euh"


def test_normalize_empty_string():
    assert _normalize("") == ""


def test_normalize_whitespace_only():
    assert _normalize("   ") == ""


def test_normalize_mid_word_hyphen_preserved():
    # Hyphens in the middle of a word are NOT stripped
    assert _normalize("well-known") == "well-known"


def test_normalize_multiple_trailing_punctuation():
    assert _normalize("really?!") == "really"


def test_normalize_nfc_leaves_already_normal_word():
    # "voilà" in NFC form should survive normalisation unchanged
    assert _normalize("Voilà") == "voilà"


def test_normalize_strips_semicolons_and_colons():
    assert _normalize("ok;") == "ok"
    assert _normalize("ok:") == "ok"
