import numpy as np

from carl_cam.features import FeatureExtractor
from carl_cam.records import HandObservation


def hand(*, offset=(300, 350), scale=1, label="Left", missing_depth=False):
    xy = np.array([
        [0, 80], [-20, 60], [-45, 45], [-65, 25], [-85, 10],
        [-30, 15], [-30, -25], [-30, -55], [-30, -85],
        [0, 0], [0, -45], [0, -80], [0, -115],
        [25, 10], [25, -35], [25, -65], [25, -95],
        [50, 25], [60, -10], [65, -35], [70, -60],
    ], dtype=float) * scale + offset
    z = np.arange(21) * 0.001 * scale
    if missing_depth:
        z[4] = np.nan
    return HandObservation(tuple(tuple(row) for row in np.column_stack((xy, z))), label, 0.99)


def test_feature_widths_and_one_hand_presence_are_explicit() -> None:
    result = FeatureExtractor().extract((hand(),))
    assert set(result.vectors) == {112, 138, 180}
    assert all(vector.shape == (width,) for width, vector in result.vectors.items())
    assert np.array_equal(result.vectors[112][-2:], [1, 0])
    assert np.array_equal(result.vectors[112][53:106], np.zeros(53))


def test_hand_shape_features_ignore_translation_and_uniform_scale() -> None:
    first = FeatureExtractor().extract((hand(),)).vectors[180]
    second = FeatureExtractor().extract((hand(offset=(800, 450), scale=2),)).vectors[180]
    assert np.allclose(first, second)


def test_unusable_depth_keeps_xy_candidates_and_small_hands_are_rejected() -> None:
    result = FeatureExtractor().extract((hand(missing_depth=True),))
    assert set(result.vectors) == {112, 138}
    assert not FeatureExtractor().extract((hand(scale=0.2),)).vectors


def test_overlap_suppresses_two_hand_command_until_three_stable_frames() -> None:
    extractor = FeatureExtractor()
    overlapping = (hand(), hand(offset=(310, 350), label="Right"))
    assert not extractor.extract(overlapping).two_hand_stable
    separated = (hand(), hand(offset=(800, 350), label="Right"))
    assert not extractor.extract(separated).two_hand_stable
    assert not extractor.extract(separated).two_hand_stable
    assert extractor.extract(separated).two_hand_stable


def test_no_hand_does_not_invent_a_feature_row() -> None:
    assert not FeatureExtractor().extract(()).vectors
