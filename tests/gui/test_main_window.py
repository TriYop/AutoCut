from autocut.gui.widgets.main_window import AutoCutMainWindow


def test_run_button_initially_disabled(qtbot):
    window = AutoCutMainWindow()
    qtbot.addWidget(window)
    assert not window.run_button.isEnabled()


def test_file_selected_enables_run_button(qtbot):
    window = AutoCutMainWindow()
    qtbot.addWidget(window)
    window.file_drop.file_selected.emit("/tmp/video.mp4")
    assert window.run_button.isEnabled()


def test_empty_file_selected_keeps_button_disabled(qtbot):
    window = AutoCutMainWindow()
    qtbot.addWidget(window)
    window.file_drop.file_selected.emit("")
    assert not window.run_button.isEnabled()


def test_pipeline_log_appended_to_log_pane(qtbot):
    window = AutoCutMainWindow()
    qtbot.addWidget(window)
    window._on_pipeline_log("Extracting audio…")
    assert "Extracting audio" in window.log.toPlainText()


def test_pipeline_done_resets_button_text_and_state(qtbot):
    window = AutoCutMainWindow()
    qtbot.addWidget(window)
    window.run_button.setText("Running…")
    window.run_button.setEnabled(False)
    window._on_pipeline_done()
    assert window.run_button.text() == "Run"
    assert window.run_button.isEnabled()


def test_pipeline_done_hides_progress_bar(qtbot):
    window = AutoCutMainWindow()
    qtbot.addWidget(window)
    window.progress_bar.setVisible(True)
    window._on_pipeline_done()
    assert not window.progress_bar.isVisible()


def test_pipeline_error_resets_button(qtbot):
    window = AutoCutMainWindow()
    qtbot.addWidget(window)
    window.run_button.setEnabled(False)
    window._on_pipeline_error("Something went wrong")
    assert window.run_button.isEnabled()
    assert window.run_button.text() == "Run"


def test_pipeline_error_shows_message_in_log(qtbot):
    window = AutoCutMainWindow()
    qtbot.addWidget(window)
    window._on_pipeline_error("No audio track found")
    assert "No audio track found" in window.log.toPlainText()


def test_pipeline_error_hides_progress_bar(qtbot):
    window = AutoCutMainWindow()
    qtbot.addWidget(window)
    window.progress_bar.setVisible(True)
    window._on_pipeline_error("oops")
    assert not window.progress_bar.isVisible()
