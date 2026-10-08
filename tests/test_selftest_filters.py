"""Installed render smoke checks must actually exercise all three recipes."""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from carl_cam import graphics, render
from carl_cam.selftest import check_filter_rendering


@pytest.fixture(scope='module')
def qapp():
    return QApplication.instance() or QApplication([])


def test_graphics_selftest_exercises_all_filters_and_fallbacks(qapp, monkeypatch):
    calls = []
    compose = render.compose_palette

    def tracked(pixels, state, **kwargs):
        calls.append((state.theme, state.reduced_motion, kwargs['masks'] is None))
        return compose(pixels, state, **kwargs)

    monkeypatch.setattr(render, 'compose_palette', tracked)
    result = check_filter_rendering()
    assert len(calls) == 12
    for theme in ('red', 'yellow', 'blue'):
        assert result[theme]['cases'] == 4
        assert (theme, True, False) in calls
        assert (theme, False, True) in calls


def test_graphics_selftest_rejects_plain_filter(qapp, monkeypatch):
    compose = render.compose_palette

    def plain(pixels, state, **kwargs):
        _, stages = compose(pixels, state, **kwargs)
        return pixels.copy(), stages

    monkeypatch.setattr(render, 'compose_palette', plain)
    with pytest.raises(AssertionError):
        check_filter_rendering()


def test_graphics_selftest_rejects_missing_border(qapp, monkeypatch):
    monkeypatch.setattr(graphics, 'draw_border', lambda pixels, *args, **kwargs: np.copy(pixels))
    with pytest.raises(AssertionError):
        check_filter_rendering()
