"""Primary PyTorch candidate fitting with session-separated validation.

This module fits candidates; it does not select a live model from frame accuracy
or activate one. The guided action exercise ranks passing candidates, followed
by one sealed check and the separate activation rehearsal.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import Event

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.nn import functional

from carl_cam.features import FEATURE_SCHEMAS
from carl_cam.storage import LABELS


@dataclass(frozen=True, slots=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> Standardizer:
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Training measurements must be finite")
        return cls(values.mean(axis=0).astype(np.float32), np.maximum(values.std(axis=0), 1e-6).astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-1] != len(self.mean) or not np.isfinite(values).all():
            raise ValueError("Model and feature schema do not match")
        return ((values - self.mean) / self.scale).astype(np.float32)


class GestureNetwork(nn.Module):
    def __init__(self, width: int, hidden: tuple[int, int]) -> None:
        super().__init__()
        if width not in FEATURE_SCHEMAS or hidden not in ((64, 32), (128, 64)):
            raise ValueError("Unsupported neural architecture")
        self.layers = nn.Sequential(
            nn.Linear(width, hidden[0]), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden[1], len(LABELS)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


@dataclass(slots=True)
class NeuralCandidate:
    width: int
    hidden: tuple[int, int]
    network: GestureNetwork
    scaler: Standardizer
    validation_loss: float
    validation_recall: tuple[float, ...]
    epochs: int
    session_ids: dict[str, tuple[str, ...]]
    threshold: float = 0.80
    margin: float = 0.15
    training_device: str = "cpu"

    @property
    def name(self) -> str:
        return f"{self.width} → {self.hidden[0]} → {self.hidden[1]} → 6"

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.network.parameters())

    def scores(self, values: np.ndarray) -> np.ndarray:
        standardized = self.scaler.transform(np.atleast_2d(values))
        self.network.eval()
        with torch.inference_mode():
            return self.network(torch.from_numpy(standardized)).softmax(dim=1).numpy()


@dataclass(frozen=True, slots=True)
class ComparisonBaseline:
    width: int
    validation_recall: tuple[float, ...]
    coefficients: np.ndarray
    intercept: np.ndarray


@dataclass(frozen=True, slots=True)
class FitResult:
    candidates: tuple[NeuralCandidate, ...]
    baselines: tuple[ComparisonBaseline, ...]
    depth_used: bool
    requested_device: str = "cpu"
    cpu_fallback: bool = False

    @property
    def backend_summary(self) -> str:
        if self.cpu_fallback:
            return "CPU fallback was used. Finished models also work on CPU."
        if self.requested_device == "cuda":
            return "Trained on NVIDIA GPU. Finished models also work on CPU."
        return "Trained on CPU."


def gpu_training_status() -> tuple[bool, str]:
    """Report training availability without starting training or changing drivers."""
    if torch.version.cuda is None:
        return False, "This build has CPU-only training. NVIDIA training needs the GPU-enabled Carl-Cam build."
    try:
        available = torch.cuda.is_available()
    except (RuntimeError, OSError):
        return False, "NVIDIA training could not initialize. CPU training is available."
    if not available:
        return False, "No compatible NVIDIA GPU and driver are available. CPU training is available."
    return True, "NVIDIA training is available. Leave this unchecked to train on CPU."


def depth_coverage(rows: Sequence[dict]) -> dict[str, tuple[int, int]]:
    """Count usable-depth fit rows without opening sealed feature values.

    Ten rows per class in each fit partition is a prototype minimum (at most
    five saved rows/second). Actual quality still requires all activation gates.
    One missing depth row does not discard the preferred feature candidate.
    """
    counts = {label: [0, 0] for label in LABELS}
    for row in rows:
        index, label = row.get("round_index"), row.get("label")
        if index == 4 or label not in LABELS:
            continue
        values = row.get("features", {}).get("180")
        if values is not None:
            vector = np.asarray(values, np.float32)
            if vector.shape == (180,) and np.isfinite(vector).all():
                counts[label][0 if index < 3 else 1] += 1
    return {label: tuple(value) for label, value in counts.items()}


@dataclass(frozen=True, slots=True)
class ActionResult:
    candidate: NeuralCandidate
    pose_successes: tuple[int, int, int, int, int]
    unintended_commands: int
    p95_delay_ms: float

    @property
    def passed(self) -> bool:
        return (
            len(self.pose_successes) == 5
            and all(type(count) is int and 4 <= count <= 5 for count in self.pose_successes)
            and self.unintended_commands == 0
            and np.isfinite(self.p95_delay_ms) and self.p95_delay_ms >= 0
        )


def select_action_winner(results: Sequence[ActionResult]) -> NeuralCandidate | None:
    passed = [result for result in results if result.passed]
    if not passed:
        return None
    best_counts = max((min(result.pose_successes), sum(result.pose_successes)) for result in passed)
    tied = [result for result in passed if (min(result.pose_successes), sum(result.pose_successes)) == best_counts]
    fastest = min(result.p95_delay_ms for result in tied)
    near = [result for result in tied if result.p95_delay_ms <= fastest + 50]
    return min(near, key=lambda result: result.candidate.parameter_count).candidate


def _recalls(targets: np.ndarray, predictions: np.ndarray) -> tuple[float, ...]:
    return tuple(float(np.mean(predictions[targets == label] == label)) for label in range(len(LABELS)))


def _partitions(rows: Sequence[dict], width: int, include_depth: bool):
    partitions = {"train": [], "validation": []}
    session_ownership = {}
    sessions = {"train": set(), "validation": set(), "sealed": set()}
    for row in rows:
        label = row.get("label")
        index = row.get("round_index")
        session = row.get("session_id")
        if label not in LABELS or type(index) is not int or not 0 <= index < 5 or not isinstance(session, str) or not session:
            raise ValueError("Dataset has invalid labels or capture-round identities")
        split = "train" if index < 3 else "validation" if index == 3 else "sealed"
        ownership = (label, index, split)
        if session in session_ownership and session_ownership[session] != ownership:
            raise ValueError("One recording session cannot span labels or data partitions")
        session_ownership[session] = ownership
        sessions[split].add(session)
        # Never read any sealed feature values during fitting or standardization.
        if split == "sealed":
            continue
        features = row.get("features", {})
        if include_depth and "180" not in features:
            continue
        values = np.asarray(features.get(str(width), []), dtype=np.float32)
        if values.shape != (width,) or not np.isfinite(values).all():
            raise ValueError("Dataset has incompatible or unusable measurements")
        partitions[split].append((values, LABELS.index(label)))
    output = {}
    for split, observations in partitions.items():
        if not observations:
            raise ValueError(f"No {split} observations")
        values = np.stack([observation[0] for observation in observations])
        targets = np.array([observation[1] for observation in observations], dtype=np.int64)
        if set(targets) != set(range(len(LABELS))):
            raise ValueError(f"Every label needs complete {split} rounds")
        output[split] = values, targets
    return output, {split: tuple(sorted(ids)) for split, ids in sessions.items()}


def _fit_one(width: int, hidden: tuple[int, int], parts, sessions, *, device: str, cancel: Event, progress, max_epochs: int) -> NeuralCandidate:
    torch.manual_seed(17)
    train_values, train_targets = parts["train"]
    validation_values, validation_targets = parts["validation"]
    scaler = Standardizer.fit(train_values)
    x = torch.from_numpy(scaler.transform(train_values)).to(device)
    y = torch.from_numpy(train_targets).to(device)
    validation_x = torch.from_numpy(scaler.transform(validation_values)).to(device)
    validation_y = torch.from_numpy(validation_targets).to(device)
    counts = np.bincount(train_targets, minlength=len(LABELS))
    weights = torch.tensor(len(train_targets) / (len(LABELS) * counts), dtype=torch.float32, device=device)
    network = GestureNetwork(width, hidden).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.01)
    best_loss, patience, best_state = float("inf"), 0, None
    generator = torch.Generator().manual_seed(17)
    for epoch in range(max_epochs):
        if cancel.is_set():
            raise InterruptedError("Training canceled; the previous model was kept")
        network.train()
        order = torch.randperm(len(x), generator=generator).to(device)
        for batch in order.split(64):
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(network(x[batch]), y[batch], weight=weights)
            loss.backward()
            # Record a CUDA fit only after data, parameters and actual gradients
            # reside on CUDA. The returned model is moved to CPU for portable
            # evaluation, saved weights and ordinary live gesture inference.
            if device == "cuda" and (
                    not x.is_cuda or not y.is_cuda
                    or any(not parameter.is_cuda or parameter.grad is None
                           or not parameter.grad.is_cuda for parameter in network.parameters())):
                raise RuntimeError("GPU training did not execute on CUDA")
            optimizer.step()
        network.eval()
        with torch.inference_mode():
            per_row = functional.cross_entropy(network(validation_x), validation_y, reduction="none")
            validation_loss = float(torch.stack([per_row[validation_y == label].mean() for label in range(len(LABELS))]).mean().cpu())
        progress(f"Training {width} measurements, {hidden[0]}/{hidden[1]} neurons: pass {epoch + 1}")
        if validation_loss <= best_loss - 0.001:
            best_loss, patience = validation_loss, 0
            best_state = copy.deepcopy(network.state_dict())
        else:
            patience += 1
        if patience >= 10:
            break
    network.load_state_dict(best_state)
    network.cpu().eval()
    with torch.inference_mode():
        predictions = network(torch.from_numpy(scaler.transform(validation_values))).argmax(dim=1).numpy()
    return NeuralCandidate(width, hidden, network, scaler, best_loss, _recalls(validation_targets, predictions), epoch + 1, sessions,
                           training_device=device)


def fit_candidates(
    rows: Sequence[dict], *, include_depth: bool, use_cuda: bool = False,
    cancel: Event | None = None, progress: Callable[[str], None] | None = None, max_epochs: int = 100,
) -> FitResult:
    if not 1 <= max_epochs <= 100:
        raise ValueError("Training budget must be within the approved 100 passes")
    cancel = cancel or Event()
    progress = progress or (lambda message: None)
    if cancel.is_set():
        raise InterruptedError("Training canceled; the previous model was kept")
    requested_device = "cuda" if use_cuda else "cpu"
    available = gpu_training_status()[0] if use_cuda else False
    device = "cuda" if available else "cpu"
    cpu_fallback = bool(use_cuda and not available)
    if cpu_fallback:
        progress("NVIDIA training is unavailable. Training on CPU instead.")
    elif device == "cuda":
        progress("Training on NVIDIA GPU. Finished models will also work on CPU.")
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        progress("Training on CPU.")
    torch.set_num_threads(2)
    candidates, baselines = [], []
    for width in (112, 138, 180) if include_depth else (112, 138):
        parts, sessions = _partitions(rows, width, include_depth)
        for hidden in ((64, 32), (128, 64)):
            try:
                candidate = _fit_one(width, hidden, parts, sessions, device=device, cancel=cancel, progress=progress, max_epochs=max_epochs)
            except RuntimeError:
                if device == "cpu":
                    raise
                progress("GPU training failed. Retrying on CPU.")
                cpu_fallback = True
                device = "cpu"
                candidate = _fit_one(width, hidden, parts, sessions, device=device, cancel=cancel, progress=progress, max_epochs=max_epochs)
            candidates.append(candidate)
        scaler = candidates[-1].scaler
        train_x, train_y = parts["train"]
        val_x, val_y = parts["validation"]
        comparison = LogisticRegression(solver="lbfgs", C=1.0, class_weight="balanced", max_iter=1000)
        comparison.fit(scaler.transform(train_x), train_y)
        baseline_predictions = comparison.predict(scaler.transform(val_x))
        baselines.append(ComparisonBaseline(width, _recalls(val_y, baseline_predictions), comparison.coef_, comparison.intercept_))
    devices = {candidate.training_device for candidate in candidates}
    progress("Training complete — " + ("NVIDIA GPU; models returned to CPU." if devices == {"cuda"}
             else "CPU." if not use_cuda else "CPU fallback was used."))
    return FitResult(tuple(candidates), tuple(baselines), include_depth, requested_device, cpu_fallback)
