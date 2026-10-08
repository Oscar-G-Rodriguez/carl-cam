import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from carl_cam.menu import perimeter
from carl_cam.render import MenuTransition, SceneState, compose_palette, draw_menu


@pytest.fixture(scope="module", autouse=True)
def menu_application():
    yield QApplication.instance() or QApplication([])


def test_menu_labels_have_identical_upright_local_pixels_in_both_orientations():
    rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    layout = perimeter()
    preview = draw_menu(rgb, layout, selected="red")
    output = draw_menu(rgb, layout.output(), selected="red")
    for (name, rect), (output_name, mirrored) in zip(layout.cards, layout.output().cards, strict=True):
        assert name == output_name
        patch = preview[int(rect.y):int(rect.y + rect.height), int(rect.x):int(rect.x + rect.width)]
        output_patch = output[int(mirrored.y):int(mirrored.y + mirrored.height), int(mirrored.x):int(mirrored.x + mirrored.width)]
        np.testing.assert_array_equal(patch, output_patch)
        assert not np.array_equal(patch, patch[:, ::-1])


def test_training_and_inspection_cannot_put_filters_into_the_output_scene():
    image = np.full((720, 1280, 3), (130, 160, 180), dtype=np.uint8)
    quad = ((300, 150), (980, 150), (980, 570), (300, 570))
    for mode in ("training", "inspect", "setup"):
        result, _ = compose_palette(image, SceneState("red", quad, mode))
        np.testing.assert_array_equal(result, image)
    normal, _ = compose_palette(image, SceneState("red", quad, "camera"))
    assert not np.array_equal(normal, image)


def test_menu_transition_is_time_based_and_private_modes_clear_immediately():
    transition = MenuTransition()
    state = SceneState(mode="menu", menu=perimeter())
    assert transition.update(state, 1)[1] == 0
    assert transition.update(state, 1.09)[1] == pytest.approx(.5)
    assert transition.update(state, 1.18)[1] == pytest.approx(1)
    assert transition.update(SceneState(), 1.3)[1] == pytest.approx(1)
    assert transition.update(SceneState(), 1.36)[1] == pytest.approx(.5)
    assert transition.update(SceneState(), 1.42)[1] == pytest.approx(0, abs=1e-9)
    reduced = SceneState(mode="menu", menu=perimeter(), reduced_motion=True)
    assert transition.update(reduced, 2)[1] == 1
    assert transition.update(SceneState(mode="training"), 2.01) == (None, 0.)
