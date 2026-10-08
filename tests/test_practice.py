from carl_cam.practice import ControlPractice

ASSIGNMENTS = {"thumbs_up": "blue", "open_palm": "red", "ily": "yellow"}
QUAD = ((300, 150), (980, 150), (980, 570), (300, 570))


def test_hand_practice_requires_real_frame_clear_direct_selection_and_menu_paths():
    flow = ControlPractice(ASSIGNMENTS, hands=True)
    assert not flow.observe(mode="camera", quad=QUAD, theme="red", manual=True)
    assert flow.observe(mode="framed", quad=QUAD, theme="red", command="frame")
    for theme in ("red", "yellow", "blue"):
        assert not flow.observe(mode="framed", quad=QUAD, theme=theme, command=f"select:{theme}")
        assert flow.observe(mode="camera", quad=None, theme=theme)
        assert "Hold " in flow.instruction
        assert not flow.observe(mode="camera", quad=None, theme=theme, command=f"select:{theme}", live=False)
        assert flow.observe(mode="camera", quad=None, theme=theme, command=f"select:{theme}")
        assert flow.observe(mode="framed", quad=QUAD, theme=theme, command="frame")
    assert flow.observe(mode="menu", quad=None, theme="blue", command="menu")
    assert not flow.observe(mode="camera", quad=None, theme="blue", command="back")
    assert flow.observe(mode="camera", quad=None, theme="yellow", command="yellow")
    assert flow.observe(mode="framed", quad=QUAD, theme="yellow", command="frame")
    assert flow.complete and "not a model accuracy test" in flow.instruction


def test_desktop_practice_is_accessible_without_a_model_or_dataset():
    flow = ControlPractice(ASSIGNMENTS, hands=False)
    assert "Mouse frame" in flow.instruction
    assert flow.observe(mode="camera", quad=QUAD, theme="red", manual=True)
    assert flow.observe(mode="camera", quad=None, theme="red")
    assert "Selected filter" in flow.instruction


def test_camera_window_practice_uses_private_qt_instructions_and_real_controls(tmp_path):
    from PySide6.QtWidgets import QApplication
    from test_ui import FakeCamera

    from carl_cam.storage import Store
    from carl_cam.ui import CameraWindow

    app = QApplication.instance() or QApplication([])
    window = CameraWindow(test_camera=FakeCamera(), store=Store(tmp_path / "CarlCam"))
    window.start_camera()
    window.start_practice()
    assert window.practice is not None and not window.practice.hands
    window.enable_mouse_frame()
    assert window.practice.index == 1
    window._set_quad(None)
    assert window.practice.index == 2
    window.themes.activated.emit(window.themes.currentIndex())
    assert window.practice.index == 3
    window.enable_mouse_frame()
    assert window.practice.index == 4
    window.end_practice()
    assert window.practice is None and window.practice_panel.isHidden()
    window.close()
    app.processEvents()
