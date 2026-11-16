"""Shared AdamW + warmup/decay-scheduler optimizer construction.

Used by both [Trainer][linkingtk.train.trainer.Trainer] and
[CrossEncoderTrainer][linkingtk.train.cross_encoder.CrossEncoderTrainer].
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from linkingtk.train.arguments import TrainingArguments

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler

_NO_DECAY = ("bias", "LayerNorm.weight", "LayerNorm.bias")


def build_optimizer(
    model: Any, args: TrainingArguments, num_training_steps: int
) -> tuple[torch.optim.Optimizer, LRScheduler | None]:
    """Build an ``AdamW`` optimizer (weight-decay-grouped) and an optional warmup scheduler.

    Bias and LayerNorm parameters never get weight decay, matching
    standard BERT fine-tuning convention regardless of `args.weight_decay`.
    With ``args.lr_scheduler == "linear"`` (the default),
    `args.warmup_ratio <= 0.0` returns ``None`` for the scheduler -- every
    caller's plain constant-LR ``AdamW`` behavior from before these fields
    existed. ``"cosine"`` always returns a cosine-annealing scheduler.
    """
    named_params = list(model.named_parameters())
    grouped = [
        {
            "params": [p for n, p in named_params if not any(nd in n for nd in _NO_DECAY)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in named_params if any(nd in n for nd in _NO_DECAY)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped, lr=args.learning_rate)
    num_warmup_steps = int(max(args.warmup_ratio, 0.0) * num_training_steps)
    if args.lr_scheduler == "cosine":
        from transformers import get_cosine_schedule_with_warmup

        cosine = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return optimizer, cosine
    if args.warmup_ratio <= 0.0:
        return optimizer, None

    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(  # type: ignore[no-untyped-call]
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler
