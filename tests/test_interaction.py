from carl_cam.interaction import InputObservation, InteractionController, winning_pose

QUAD = ((300, 150), (980, 150), (980, 570), (300, 570))


def observation(time, label, **values):
    scores = [0.01] * 6
    scores[label] = 0.95
    return InputObservation(time, tuple(scores), **values)


def test_frame_requires_dwell_and_expires_after_tracking_grace() -> None:
    controller = InteractionController()
    for time in (1.0, 1.1, 1.2, 1.31):
        result = controller.update(observation(time, 0, quad=QUAD, two_hand_stable=True), time)
    assert result.mode == "framed" and result.quad == QUAD
    assert controller.update(None, 1.4).quad == QUAD
    assert controller.update(None, 1.47).quad is None


def test_uncertain_and_stale_predictions_cannot_start_commands() -> None:
    assert winning_pose((0.45, 0.44, 0, 0, 0, 0)) is None
    controller = InteractionController()
    for time in (1.0, 1.3, 1.6):
        assert controller.update(observation(time - 0.2, 1), time).mode == "camera"


def test_existing_quad_uses_100ms_position_window_but_new_commands_need_80ms():
    controller = InteractionController()
    for time in (1., 1.1, 1.2, 1.31):
        controller.update(observation(time, 0, quad=QUAD, two_hand_stable=True), time)
    moved = tuple((x + 15, y) for x, y in QUAD)
    result = controller.update(observation(1.36, 0, quad=moved, two_hand_stable=True), 1.45)
    assert result.mode == "framed" and result.quad == moved and result.command is None
    fresh_controller = InteractionController()
    for time in (2., 2.4, 2.8):
        result = fresh_controller.update(observation(time - .09, 0, quad=QUAD, two_hand_stable=True), time)
    assert result.mode == "camera" and result.quad is None


def test_direct_selection_fires_once_until_release_and_training_dispatches_nothing() -> None:
    controller = InteractionController(assignments={"thumbs_up": "blue", "open_palm": "yellow", "ily": "red"})
    controller.update(observation(1, 2), 1)
    assert controller.update(observation(1.31, 2), 1.31).command == "select:blue"
    assert controller.update(observation(2, 2), 2).command is None
    controller.enter_mode("training")
    assert controller.update(observation(3, 1), 3).command is None


def test_pinch_must_start_and_end_over_same_target_and_tracking_loss_cancels() -> None:
    controller = InteractionController()
    controller.dispatch("menu")
    for time in (1.0, 1.11):
        controller.update(InputObservation(time, pointer=(900, 300), pinch_ratio=0.8, hover="blue"), time)
    controller.update(InputObservation(1.2, pointer=(900, 300), pinch_ratio=0.2, hover="blue"), 1.2)
    result = controller.update(InputObservation(1.3, pointer=(900, 300), pinch_ratio=0.8, hover="blue"), 1.3)
    assert result.mode == "camera" and result.theme == "blue" and result.command == "blue"
    controller.dispatch("menu")
    for time in (2.0, 2.11):
        controller.update(InputObservation(time, pointer=(100, 300), pinch_ratio=0.8, hover="red"), time)
    controller.update(InputObservation(2.2, pointer=(100, 300), pinch_ratio=0.2, hover="red"), 2.2)
    controller.update(None, 2.4)
    result = controller.update(InputObservation(2.5, pointer=(100, 300), pinch_ratio=0.8, hover="red"), 2.5)
    assert result.mode == "menu" and result.theme == "blue"
