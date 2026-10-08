import pytest

from carl_cam.performance import PerformanceFallback


def feed(policy, *, start=0, end=5, fps=20, delay=100, level=0, enabled=True):
    results = []
    for index in range(round((end - start) * 2) + 1):
        now = start + index / 2
        result = policy.observe(now, int(now * fps), delay, level, enabled=enabled)
        if result is not None:
            results.append((now, result))
    return results


def test_throughput_requires_full_window_and_counts_renders_not_source_ids():
    policy = PerformanceFallback()
    assert feed(policy, end=4.5) == []
    assert policy.observe(5, 100, 100, 0) == 1


def test_latency_uses_strict_threshold_and_never_upgrades():
    assert feed(PerformanceFallback(), fps=30, delay=150) == []
    assert feed(PerformanceFallback(), fps=30, delay=151) == [(5, 1)]
    assert feed(PerformanceFallback(), fps=27) == []
    assert feed(PerformanceFallback(), level=2) == []
    assert feed(PerformanceFallback(), fps=30, level=1) == []


def test_fallback_cooldown_survives_private_modes_and_waits_ten_seconds():
    policy = PerformanceFallback()
    assert feed(policy) == [(5, 1)]
    policy.observe(5.1, 100, 100, 1, enabled=False)
    assert feed(policy, start=5.5, end=14.5, level=1) == []
    assert policy.observe(15, 300, 100, 1) == 2


def test_private_modes_discard_partial_window_and_no_output_is_not_30fps():
    policy = PerformanceFallback()
    feed(policy, end=4)
    policy.observe(4.1, 82, None, 0, enabled=False)
    assert feed(policy, start=4.5, end=9) == []
    assert policy.observe(9.5, 190, 100, 0) == 1
    assert feed(PerformanceFallback(), fps=0, delay=None) == [(5, 1)]


def test_duplicate_delays_do_not_dominate_frame_median_and_history_is_bounded():
    policy = PerformanceFallback()
    for index in range(50):
        policy.observe(index / 100, 0, 300, 0)
    for index in range(1, 151):
        policy.observe(.5 + index * 4.5 / 150, index, 100, 0)
    assert policy.last_change == float('-inf')
    for index in range(1000):
        policy.observe(10 + index / 1000, 151, 100, 0)
    assert len(policy.samples) <= 512


@pytest.mark.parametrize('sequence,delay,level', [(True, 100, 0), (-1, 100, 0),
                                                (1, float('nan'), 0), (1, 100, 3)])
def test_bad_measurements_do_not_trigger_quality_changes(sequence, delay, level):
    policy = PerformanceFallback()
    assert policy.observe(5, sequence, delay, level) is None
    assert not policy.samples
