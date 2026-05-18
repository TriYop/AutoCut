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
