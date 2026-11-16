"""Unit tests for `build_optimizer`'s scheduler selection."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from linkingtk.train._optim import build_optimizer  # noqa: E402
from linkingtk.train.arguments import TrainingArguments  # noqa: E402


def _lrs(args: TrainingArguments, steps: int = 10) -> list[float]:
    model = torch.nn.Linear(2, 2)
    optimizer, scheduler = build_optimizer(model, args, steps)
    lrs = []
    for _ in range(steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return lrs


def test_linear_without_warmup_has_no_scheduler() -> None:
    model = torch.nn.Linear(2, 2)
    _optimizer, scheduler = build_optimizer(model, TrainingArguments(output_dir="x"), 10)
    assert scheduler is None


def test_cosine_anneals_even_without_warmup() -> None:
    lrs = _lrs(TrainingArguments(output_dir="x", learning_rate=1.0, lr_scheduler="cosine"))
    assert lrs[0] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(lrs, lrs[1:], strict=False))
    assert lrs[-1] < 0.05


def test_cosine_with_warmup_ramps_up_first() -> None:
    lrs = _lrs(
        TrainingArguments(
            output_dir="x", learning_rate=1.0, lr_scheduler="cosine", warmup_ratio=0.2
        )
    )
    assert lrs[0] == pytest.approx(0.0)
    assert lrs[2] == pytest.approx(1.0)
