import numpy as np
import pytest
from PySide6.QtGui import QFont, QRawFont
from PySide6.QtWidgets import QApplication

from carl_cam.graphics import asset_pixels, draw_border, menu_card, place_asset, verified_svg
from carl_cam.typography import configure_typography


@pytest.fixture(scope="module")
def application():
    app = QApplication.instance() or QApplication([])
    configure_typography()
    yield app


def test_bundled_font_has_real_glyphs_when_system_database_is_unavailable(application):
    font = QRawFont.fromFont(QFont("Arimo", 11))
    assert font.isValid() and font.familyName() == "Arimo"
    assert all(font.supportsCharacter(char) for char in "Carl-Cam training 0123456789")
    assert all(font.glyphIndexesForString("Carl-Cam"))


def test_original_borders_are_distinct_transparent_read_only_vectors():
    images = [asset_pixels("border-" + theme) for theme in ("red", "yellow", "blue")]
    for image in images:
        assert image.shape == (360, 640, 4)
        assert not image.flags.writeable
        assert np.all(image[70:285, 70:565, 3] == 0)
    assert not np.array_equal(images[0], images[1])
    assert not np.array_equal(images[1], images[2])
    # Yellow preserves four pink circles, structural yellow band, and plum ink.
    yellow = images[1]
    for x, y in ((32, 32), (608, 32), (608, 328), (32, 328)):
        np.testing.assert_array_equal(yellow[y, x, :3], (241, 106, 165))
    assert np.any(np.all(yellow[..., :3] == (37, 22, 47), axis=-1) & (yellow[..., 3] > 0))


def test_blue_corner_brackets_open_toward_interior_on_both_axes():
    blue = asset_pixels("border-blue")
    # All four outward tips are filled; inward diagonal cutouts stay clear.
    # Reusing the top-left polygon at every corner fails this check.
    for x, y in ((20, 20), (620, 20), (620, 340), (20, 340)):
        assert blue[y, x, 3] == 255
    for x, y in ((40, 40), (600, 40), (600, 320), (40, 320)):
        assert blue[y, x, 3] == 0


def test_pop_border_has_ten_soft_scallops_and_flat_cream_keyline():
    yellow = asset_pixels("border-yellow")
    top = yellow[14, 60:580, 3] > 0
    starts = top & ~np.r_[False, top[:-1]]
    assert np.count_nonzero(starts) == 10
    np.testing.assert_array_equal(yellow[40, 200], (255, 244, 200, 255))


def test_border_follows_tilted_quad_without_warping_interior_and_supports_hand_occlusion():
    rgb = np.full((720, 1280, 3), (90, 130, 150), np.uint8)
    quad = ((310, 150), (960, 180), (990, 590), (280, 550))
    result = draw_border(rgb, "yellow", quad)
    np.testing.assert_array_equal(result[250:450, 400:850], rgb[250:450, 400:850])
    np.testing.assert_array_equal(result[:, :200], rgb[:, :200])
    assert not np.array_equal(result, rgb)
    np.testing.assert_array_equal(draw_border(rgb, "red", quad, np.ones((720, 1280))), rgb)
    np.testing.assert_array_equal(draw_border(rgb, "red", None), rgb)


def test_menu_uses_exact_theme_exports_with_distinct_upright_text(application):
    cards = [menu_card(theme) for theme in ("red", "yellow", "blue")]
    for card in cards:
        assert card.shape == (100, 300, 4) and not card.flags.writeable
        assert np.count_nonzero(card[24:56, 24:220, 0] > 200) > 100
    assert not np.array_equal(cards[0][24:56], cards[1][24:56])
    for theme, card in zip(("red", "yellow", "blue"), cards, strict=True):
        np.testing.assert_array_equal(card, asset_pixels(f"menu-{theme}-default"))
    rgb = np.full((20, 20, 3), 120, np.uint8)
    clipped = place_asset(rgb, asset_pixels("pointer"), -16, -16)
    assert clipped.shape == rgb.shape and not np.array_equal(clipped, rgb)


@pytest.mark.parametrize("theme", ("red", "yellow", "blue"))
def test_menu_states_use_genuine_vectors_and_keep_selection_on_hover(theme):
    for interaction in ("hover", "focused"):
        np.testing.assert_array_equal(menu_card(theme, interaction=interaction),
                                      asset_pixels(f"menu-{theme}-{interaction}"))
        selected = menu_card(theme, True, interaction=interaction)
        # Hover/focus must not erase the Figma current-filter dot or labels.
        np.testing.assert_array_equal(selected[20:32, 254:266],
                                      asset_pixels(f"menu-{theme}-selected")[20:32, 254:266])
        np.testing.assert_array_equal(selected[25:54, 24:200],
                                      asset_pixels(f"menu-{theme}-selected")[25:54, 24:200])
    np.testing.assert_array_equal(menu_card(theme, True), asset_pixels(f"menu-{theme}-selected"))


def test_menu_rejects_unavailable_states():
    with pytest.raises(ValueError, match="interaction"):
        menu_card("red", interaction="disabled")


def test_offline_svg_checksum_rejects_modified_export(tmp_path, monkeypatch):
    import json

    from carl_cam import graphics
    source = verified_svg("pointer.svg")
    manifest = json.loads((graphics.ROOT / "manifest.json").read_text())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "pointer.svg").write_bytes(source + b"\n")
    monkeypatch.setattr(graphics, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        verified_svg("pointer.svg")


def test_saturated_blend_matches_nonnegative_premultiplied_source_over():
    from carl_cam.graphics import _over

    rng = np.random.default_rng(17)
    rgb = rng.integers(0, 256, (120, 200, 3), np.uint8)
    overlay = rng.random((120, 200, 4), dtype=np.float32)
    overlay[..., :3] *= 255 * overlay[..., 3:4]
    reference = np.rint(overlay[..., :3] + rgb * (1 - overlay[..., 3:4])).clip(0, 255).astype(np.uint8)
    np.testing.assert_array_equal(_over(rgb, overlay), reference)


def test_red_layer_motion_is_bounded_elapsed_time_and_keeps_corner_anchors_static():
    from carl_cam.graphics import paper_offsets
    for seconds in np.linspace(0, 4, 101):
        offsets = paper_offsets(seconds)
        assert offsets[0] == offsets[3] == (0, 0)
        assert all(np.linalg.norm(offset) <= 4 + 1e-9 for offset in offsets)
        assert paper_offsets(seconds, reduced_motion=True) == ((0, 0),) * 4
    rgb = np.full((720, 1280, 3), 120, np.uint8)
    quad = ((310, 150), (960, 180), (990, 590), (280, 550))
    first = draw_border(rgb, "red", quad, seconds=0)
    moved = draw_border(rgb, "red", quad, seconds=1)
    assert not np.array_equal(first, moved)
    np.testing.assert_array_equal(first[250:450, 400:850], rgb[250:450, 400:850])
    np.testing.assert_array_equal(draw_border(rgb, "red", quad, seconds=0, reduced_motion=True),
                                  draw_border(rgb, "red", quad, seconds=1, reduced_motion=True))
