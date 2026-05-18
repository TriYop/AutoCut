from autocut.gui.widgets.file_drop import FileDropWidget


def test_initial_path_is_empty(qtbot):
    widget = FileDropWidget()
    qtbot.addWidget(widget)
    assert widget.path == ""


def test_set_path_updates_property(qtbot):
    widget = FileDropWidget()
    qtbot.addWidget(widget)
    widget._set_path("/tmp/video.mp4")
    assert widget.path == "/tmp/video.mp4"


def test_set_path_shows_filename_only(qtbot):
    widget = FileDropWidget()
    qtbot.addWidget(widget)
    widget._set_path("/some/long/path/my_video.mp4")
    assert widget._label.text() == "my_video.mp4"


def test_set_path_emits_file_selected(qtbot):
    widget = FileDropWidget()
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.file_selected) as blocker:
        widget._set_path("/tmp/video.mp4")
    assert blocker.args[0] == "/tmp/video.mp4"


def test_second_selection_overwrites_first(qtbot):
    widget = FileDropWidget()
    qtbot.addWidget(widget)
    widget._set_path("/tmp/a.mp4")
    widget._set_path("/tmp/b.mp4")
    assert widget.path == "/tmp/b.mp4"
