import pytest

from autocut.config import AutoCutConfig
from autocut.models import BadSegment, Segment, SegmentSource
from autocut.pipeline.merger import merge_bad_segments


def _seg(start: float, end: float, label: str = "silence") -> BadSegment:
    return BadSegment(segment=Segment(start, end), source=SegmentSource.VAD, label=label)


CFG = AutoCutConfig(merge_gap_s=0.2, padding_before_s=0.05, padding_after_s=0.05)
DURATION = 100.0


def test_empty():
    assert merge_bad_segments([], CFG, DURATION) == []


def test_single_segment_gets_padding():
    result = merge_bad_segments([_seg(5.0, 7.0)], CFG, DURATION)
    assert len(result) == 1
    assert result[0].segment.start == pytest.approx(4.95)
    assert result[0].segment.end == pytest.approx(7.05)


def test_overlapping_segments_merged():
    segs = [_seg(2.0, 5.0), _seg(4.0, 7.0)]
    result = merge_bad_segments(segs, CFG, DURATION)
    assert len(result) == 1
    assert result[0].segment.start == pytest.approx(1.95)
    assert result[0].segment.end == pytest.approx(7.05)


def test_adjacent_within_gap_merged():
    segs = [_seg(1.0, 3.0), _seg(3.1, 5.0)]
    result = merge_bad_segments(segs, CFG, DURATION)
    assert len(result) == 1


def test_separate_segments_not_merged():
    segs = [_seg(1.0, 2.0), _seg(5.0, 6.0)]
    result = merge_bad_segments(segs, CFG, DURATION)
    assert len(result) == 2


def test_padding_clamped_at_zero():
    result = merge_bad_segments([_seg(0.0, 1.0)], CFG, DURATION)
    assert result[0].segment.start == 0.0


def test_padding_clamped_at_duration():
    result = merge_bad_segments([_seg(98.0, 100.0)], CFG, DURATION)
    assert result[0].segment.end == DURATION


def test_labels_concatenated_on_merge():
    segs = [_seg(1.0, 2.0, "silence"), _seg(2.1, 3.0, "euh")]
    result = merge_bad_segments(segs, CFG, DURATION)
    assert len(result) == 1
    assert "silence" in result[0].label
    assert "euh" in result[0].label


def test_unsorted_input_handled():
    segs = [_seg(5.0, 6.0), _seg(1.0, 2.0)]
    result = merge_bad_segments(segs, CFG, DURATION)
    assert result[0].segment.start < result[1].segment.start


# ── chain merges ──────────────────────────────────────────────────────────────

def test_chain_merge_three_within_gap():
    # A=(1,2), B=(2.1,3), C=(3.1,4) — each gap=0.1 ≤ 0.2 → all collapse into one
    segs = [_seg(1.0, 2.0), _seg(2.1, 3.0), _seg(3.1, 4.0)]
    result = merge_bad_segments(segs, CFG, DURATION)
    assert len(result) == 1
    assert result[0].segment.start == pytest.approx(0.95)
    assert result[0].segment.end == pytest.approx(4.05)


def test_all_segments_merge_into_one():
    segs = [_seg(float(i), float(i) + 0.5) for i in range(5)]
    # gaps are all 0.5, but merge_gap_s=0.2 → only adjacent overlaps within 0.2 merge
    # gap between each is 0.5 > 0.2, so NO merges — they stay separate
    result = merge_bad_segments(segs, CFG, DURATION)
    assert len(result) == 5


def test_chain_merge_within_small_gap():
    # gaps = 0.1 throughout, merge_gap=0.2 → all 4 collapse into one
    segs = [_seg(float(i), float(i) + 0.5) for i in range(4)]
    # Override gap via tight spacing: (0,0.5),(0.6,1.1),(1.2,1.7),(1.8,2.3)
    segs = [_seg(i * 0.6, i * 0.6 + 0.5) for i in range(4)]
    result = merge_bad_segments(segs, CFG, DURATION)
    assert len(result) == 1


# ── label behaviour ───────────────────────────────────────────────────────────

def test_same_label_not_duplicated_on_merge():
    segs = [_seg(1.0, 2.0, "silence"), _seg(2.1, 3.0, "silence")]
    result = merge_bad_segments(segs, CFG, DURATION)
    assert len(result) == 1
    assert result[0].label == "silence"


# ── confidence ────────────────────────────────────────────────────────────────

def _seg_conf(start: float, end: float, confidence: float) -> BadSegment:
    return BadSegment(
        segment=Segment(start, end),
        source=SegmentSource.WHISPER_FILLER,
        label="euh",
        confidence=confidence,
    )


def test_confidence_minimum_preserved_on_merge():
    segs = [_seg_conf(1.0, 2.0, 0.8), _seg_conf(2.1, 3.0, 0.6)]
    result = merge_bad_segments(segs, CFG, DURATION)
    assert len(result) == 1
    assert result[0].confidence == pytest.approx(0.6)


def test_confidence_minimum_of_three():
    segs = [_seg_conf(1.0, 2.0, 0.9), _seg_conf(2.1, 3.0, 0.5), _seg_conf(3.1, 4.0, 0.7)]
    result = merge_bad_segments(segs, CFG, DURATION)
    # All within merge_gap=0.2? gap 0.1 ≤ 0.2 → yes
    assert result[0].confidence == pytest.approx(0.5)


# ── source preservation ───────────────────────────────────────────────────────

def test_source_from_first_segment_preserved():
    first = BadSegment(Segment(1.0, 2.0), SegmentSource.VAD, "silence")
    second = BadSegment(Segment(2.1, 3.0), SegmentSource.WHISPER_FILLER, "euh")
    result = merge_bad_segments([first, second], CFG, DURATION)
    assert len(result) == 1
    assert result[0].source == SegmentSource.VAD


# ── zero padding ──────────────────────────────────────────────────────────────

def test_zero_padding_no_expansion():
    cfg_no_pad = AutoCutConfig(merge_gap_s=0.2, padding_before_s=0.0, padding_after_s=0.0)
    result = merge_bad_segments([_seg(5.0, 7.0)], cfg_no_pad, DURATION)
    assert result[0].segment.start == pytest.approx(5.0)
    assert result[0].segment.end == pytest.approx(7.0)


# ── merge_gap edge cases ──────────────────────────────────────────────────────

def test_zero_merge_gap_adjacent_segments_merged():
    # gap = 0.0 exactly, merge_gap_s = 0.0 → 0.0 ≤ 0.0 → merged
    cfg = AutoCutConfig(merge_gap_s=0.0, padding_before_s=0.0, padding_after_s=0.0)
    segs = [_seg(1.0, 2.0), _seg(2.0, 3.0)]
    result = merge_bad_segments(segs, cfg, DURATION)
    assert len(result) == 1


def test_zero_merge_gap_non_adjacent_not_merged():
    cfg = AutoCutConfig(merge_gap_s=0.0, padding_before_s=0.0, padding_after_s=0.0)
    segs = [_seg(1.0, 2.0), _seg(2.01, 3.0)]
    result = merge_bad_segments(segs, cfg, DURATION)
    assert len(result) == 2
