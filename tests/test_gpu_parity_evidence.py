"""Generated numerical/profile fixtures; no runtime, GPU or camera execution."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'gpu_parity_tool', Path(__file__).parents[1] / 'tools/verify_gpu_parity.py')
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)
CRITERIA = {'max_absolute_error': .02, 'p99_absolute_error': .001,
            'category_disagreement_fraction': .01}


def masks():
    result = np.zeros((360, 640, 6), np.float32)
    result[..., 0] = 1.
    return result


def test_generated_inputs_are_deterministic_rgb_and_distinct():
    first, second = parity.fixtures(), parity.fixtures()
    assert set(first) == {'black', 'white', 'gradient', 'checker', 'noise'}
    hashes = set()
    for name, image in first.items():
        assert image.shape == (360, 640, 3) and image.dtype == np.uint8
        np.testing.assert_array_equal(image, second[name])
        hashes.add(parity.hashlib.sha256(image.tobytes()).hexdigest())
    assert len(hashes) == 5


def test_identical_confidence_arrays_pass_with_zero_error():
    result = parity.compare_masks(masks(), masks(), CRITERIA)
    assert result['passed'] and result['max_absolute_error'] == 0
    assert result['category_disagreement_fraction'] == 0


def test_matching_labels_do_not_hide_excess_confidence_error():
    cpu, gpu = masks(), masks()
    gpu[..., 0], gpu[..., 1] = .995, .005
    result = parity.compare_masks(cpu, gpu, CRITERIA)
    assert result['category_disagreement_fraction'] == 0
    assert not result['passed']


def test_swapped_categories_are_counted_as_disagreements():
    gpu = masks()[..., [1, 0, 2, 3, 4, 5]]
    result = parity.compare_masks(masks(), gpu, CRITERIA)
    assert result['category_disagreement_fraction'] == 1 and not result['passed']


@pytest.mark.parametrize('invalid', [None, np.zeros((1, 6), np.float32),
                                    np.zeros((360, 640, 6), np.float16),
                                    np.zeros((360, 640, 6), np.float32),
                                    np.full((360, 640, 6), np.nan, np.float32),
                                    np.full((360, 640, 6), np.inf, np.float32),
                                    np.full((360, 640, 6), -.1, np.float32),
                                    np.full((360, 640, 6), 1.1, np.float32)])
def test_invalid_confidence_evidence_is_rejected(invalid):
    with pytest.raises(ValueError, match='confidence'):
        parity.compare_masks(masks(), invalid, CRITERIA)


def test_operator_evidence_counts_cuda_nodes_not_provider_registration():
    events = [{'cat': 'Session', 'args': {}},
              {'cat': 'Node', 'args': {'provider': 'CUDAExecutionProvider'}},
              {'cat': 'Node', 'args': {'provider': 'CUDAExecutionProvider'}}]
    assert parity.cuda_profile_counts(events) == {'CUDAExecutionProvider': 2}


@pytest.mark.parametrize('events', [[], {}, [None], [{'cat': 'Node', 'args': None}],
                                  [{'cat': 'Node', 'args': {'provider': []}}],
                                  [{'cat': 'Session', 'args': {'provider': 'CUDAExecutionProvider'}}],
                                  [{'cat': 'Node', 'args': {'provider': 'CPUExecutionProvider'}}],
                                  [{'cat': 'Node', 'args': {'provider': 'CUDAExecutionProvider'}},
                                   {'cat': 'Node', 'args': {'provider': 'CPUExecutionProvider'}}]])
def test_missing_or_cpu_operator_evidence_cannot_pass(events):
    with pytest.raises((ValueError, TypeError)):
        parity.cuda_profile_counts(events)
