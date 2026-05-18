import pytest

from autocut.config import AutoCutConfig
from autocut.gui.widgets.params_panel import ParamsPanel


@pytest.fixture
def panel(qtbot):
    p = ParamsPanel()
    qtbot.addWidget(p)
    return p


def test_default_model_matches_config(panel):
    assert panel.as_config().whisper_model == AutoCutConfig().whisper_model


def test_default_min_silence_matches_config(panel):
    assert panel.as_config().vad_min_silence_duration_ms == AutoCutConfig().vad_min_silence_duration_ms


def test_default_merge_gap_matches_config(panel):
    assert panel.as_config().merge_gap_s == pytest.approx(AutoCutConfig().merge_gap_s)


def test_default_fillers_match_config(panel):
    assert panel.as_config().filler_words == AutoCutConfig().filler_words


def test_model_change_reflected_in_config(panel):
    panel.model.setCurrentText("large-v3")
    assert panel.as_config().whisper_model == "large-v3"


def test_min_silence_change_reflected_in_config(panel):
    panel.min_silence_ms.setValue(400)
    assert panel.as_config().vad_min_silence_duration_ms == 400


def test_custom_fillers_parsed_correctly(panel):
    panel.fillers.setText("euh, bah, hein")
    assert panel.as_config().filler_words == ["euh", "bah", "hein"]


def test_empty_fillers_falls_back_to_defaults(panel):
    panel.fillers.setText("   ")
    assert panel.as_config().filler_words == AutoCutConfig().filler_words


def test_no_silence_cap_returns_none_duration(panel):
    panel.no_silence_cap.setChecked(True)
    assert panel.as_config().vad_max_silence_duration_s is None


def test_no_silence_cap_disables_max_silence_spinbox(panel):
    assert panel.max_silence_s.isEnabled()
    panel.no_silence_cap.setChecked(True)
    assert not panel.max_silence_s.isEnabled()


def test_room_eq_checkbox_enables_gain_spinbox(panel):
    assert not panel.room_eq_gain.isEnabled()
    panel.room_eq.setChecked(True)
    assert panel.room_eq_gain.isEnabled()


def test_room_eq_reflected_in_config(panel):
    panel.room_eq.setChecked(True)
    panel.room_eq_gain.setValue(-15.0)
    config = panel.as_config()
    assert config.room_eq_enabled is True
    assert config.room_eq_gain_db == pytest.approx(-15.0)


def test_no_repetitions_reflected_in_config(panel):
    panel.no_repetitions.setChecked(True)
    assert panel.as_config().detect_repetitions is False


def test_output_mode_value_property(panel):
    panel.output_mode.setCurrentText("both")
    assert panel.output_mode_value == "both"


def test_output_dir_value_empty_by_default(panel):
    assert panel.output_dir_value == ""


def test_verbose_value_false_by_default(panel):
    assert panel.verbose_value is False


def test_language_none_when_blank(panel):
    panel.language.setText("")
    assert panel.as_config().whisper_language is None


def test_language_set_when_provided(panel):
    panel.language.setText("fr")
    assert panel.as_config().whisper_language == "fr"
