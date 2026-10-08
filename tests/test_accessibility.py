from carl_cam.accessibility import (
    CONTROL_DISABLED_BORDER,
    CONTROL_DISABLED_TEXT,
    CONTROL_FOCUS_STROKE,
    CONTROL_PADDING_VERTICAL,
    CONTROL_RADIUS,
    desktop_style,
)


def test_desktop_controls_match_shared_figma_foundations():
    style = desktop_style()
    assert CONTROL_RADIUS == 8
    assert f"border-radius: {CONTROL_RADIUS}px" in style
    assert f"padding: {CONTROL_PADDING_VERTICAL}px 8px" in style
    assert f"border: {CONTROL_FOCUS_STROKE}px solid #F5C915" in style
    assert f"color: {CONTROL_DISABLED_TEXT}" in style
    assert f"border-color: {CONTROL_DISABLED_BORDER}" in style


def test_high_contrast_keeps_visible_interaction_states():
    style = desktop_style(True)
    assert "background-color: #000000; color: #FFFFFF" in style
    assert "QPushButton:hover" in style and "border-color: #F16AA5" in style
    assert "QPushButton:focus" in style and "solid #F5C915" in style
    assert "QPushButton:disabled" in style


def test_verified_checkbox_tick_and_native_missing_asset_fallback(monkeypatch):
    style = desktop_style()
    assert 'QCheckBox::indicator:checked' in style and 'icon-check.svg' in style
    assert 'width: 14px; height: 14px' in style and 'border-radius: 4px' in style

    def missing(*args):
        raise ValueError('Fixture export unavailable')

    monkeypatch.setattr('carl_cam.accessibility.verified_svg', missing)
    fallback = desktop_style()
    assert 'QCheckBox::indicator' not in fallback  # Let Qt retain its native mark.
    assert 'QCheckBox:focus' in fallback and 'min-height: 44px' in fallback
