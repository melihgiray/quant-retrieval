"""Fine-tune the encoder with in-batch contrastive learning.

The loop is deliberately plain: one optimiser, one schedule, one loss, no
framework in between. Every choice that is not obvious is written down next to
the code that makes it.

The batch is the training signal here. Each pair's positive is its own answer and
its negatives are the other answers in the same batch, so a bigger batch is a
harder and more informative problem. That is also why the loader drops the last
partial batch instead of padding it, and why gradient accumulation is not offered:
accumulation averages gradients across steps, it does not put more negatives in
front of any single one, so it makes the loss cheaper without making it better.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from quant_retrieval.models.dataset import PairCollator, TrainingPair
from quant_retrieval.models.encoder import TextEncoder, parameter_groups
from quant_retrieval.models.loss import (
    in_batch_accuracy,
    info_nce_loss,
    info_nce_with_negatives,
)
from quant_retrieval.models.schedule import linear_warmup_decay
from quant_retrieval.runtime import choose_device, set_seed

STATE_FILE = "training_state.pt"


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 256
    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    temperature: float = 0.05
    seed: int = 17
    limit_pairs: int | None = None
    device: str = "auto"
    log_every: int = 20
    pooling: str = "mean"
    # Mined wrong answers attached to each question. Zero means in-batch only.
    negatives_per_query: int = 0
    # Uniformly drawn documents added on top of the mined ones. Zero reproduces
    # the mined-only behaviour exactly.
    random_negatives_per_query: int = 0
    negatives_path: str | None = None
    # Recompute activations in the backward pass instead of storing them.
    # Roughly 30 percent slower and several times smaller in memory, and unlike
    # gradient accumulation it is mathematically the same run: same batch, same
    # negatives, same gradients. Only the timings stop being comparable.
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2 for in-batch negatives")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if (self.negatives_per_query or self.random_negatives_per_query) and (
            not self.negatives_path
        ):
            raise ValueError("negatives_per_query needs negatives_path")


@dataclass
class TrainingHistory:
    steps: list[dict[str, Any]] = field(default_factory=list)
    epochs: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"steps": self.steps, "epochs": self.epochs}


def build_dataloader(
    pairs: Sequence[TrainingPair], config: TrainingConfig, tokenizer
) -> DataLoader:
    if len(pairs) < config.batch_size:
        raise ValueError(
            f"{len(pairs)} pairs is fewer than one batch of {config.batch_size}"
        )
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        list(pairs),
        batch_size=config.batch_size,
        shuffle=True,
        # A trailing batch of one would make in-batch negatives meaningless and
        # the loss undefined, so it is dropped rather than special cased.
        drop_last=True,
        collate_fn=PairCollator(tokenizer, config.max_length),
        generator=generator,
        # Tokenising in the main process. Worker processes deadlock against the
        # fast tokenizer on macOS often enough that the throughput is not worth it.
        num_workers=0,
    )


def train(
    config: TrainingConfig,
    pairs: Sequence[TrainingPair],
    *,
    resume: bool = False,
    on_log: Callable[[str], None] = print,
) -> TrainingHistory:
    """Fine-tune and write one checkpoint per epoch. Returns the loss history."""
    set_seed(config.seed)
    device = choose_device(config.device)

    if config.limit_pairs is not None:
        pairs = list(pairs)[: config.limit_pairs]

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    encoder = TextEncoder(config.model_name, pooling=config.pooling).to(device)
    if config.gradient_checkpointing:
        encoder.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    loader = build_dataloader(pairs, config, tokenizer)

    total_steps = len(loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    optimizer = AdamW(
        parameter_groups(encoder, config.weight_decay), lr=config.learning_rate
    )
    scheduler = LambdaLR(
        optimizer, lambda step: linear_warmup_decay(step, total_steps, warmup_steps)
    )

    history = TrainingHistory()
    first_epoch = 0
    state_path = config.output_dir / STATE_FILE
    if resume and state_path.exists():
        first_epoch = restore_training_state(
            state_path, encoder, optimizer, scheduler, history, device
        )
        on_log(f"resumed after epoch {first_epoch}")

    on_log(
        f"{len(pairs)} pairs, {len(loader)} steps per epoch, {total_steps} total, "
        f"{warmup_steps} warmup, device {device}"
    )

    global_step = first_epoch * len(loader)
    for epoch in range(first_epoch, config.epochs):
        encoder.train()
        epoch_started = time.perf_counter()
        losses: list[float] = []
        accuracies: list[float] = []

        for batch in loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            query_embeddings = encoder(batch["query_input_ids"], batch["query_attention_mask"])
            document_embeddings = encoder(
                batch["document_input_ids"], batch["document_attention_mask"]
            )
            if "negative_input_ids" in batch:
                flat = encoder(batch["negative_input_ids"], batch["negative_attention_mask"])
                negative_embeddings = flat.reshape(
                    query_embeddings.shape[0], -1, flat.shape[-1]
                )
                loss = info_nce_with_negatives(
                    query_embeddings,
                    document_embeddings,
                    negative_embeddings,
                    temperature=config.temperature,
                )
            else:
                loss = info_nce_loss(
                    query_embeddings, document_embeddings, temperature=config.temperature
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                accuracy = in_batch_accuracy(query_embeddings, document_embeddings)
            losses.append(loss.item())
            accuracies.append(accuracy)
            global_step += 1

            if global_step % config.log_every == 0:
                entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": round(float(loss.item()), 4),
                    "in_batch_accuracy": round(accuracy, 4),
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                history.steps.append(entry)
                on_log(
                    f"step {global_step}/{total_steps} loss {entry['loss']:.4f} "
                    f"acc {entry['in_batch_accuracy']:.3f}"
                )

        summary = {
            "epoch": epoch,
            "mean_loss": round(sum(losses) / len(losses), 4),
            "mean_in_batch_accuracy": round(sum(accuracies) / len(accuracies), 4),
            "seconds": round(time.perf_counter() - epoch_started, 1),
        }
        history.epochs.append(summary)
        on_log(
            f"epoch {epoch} mean loss {summary['mean_loss']:.4f} "
            f"acc {summary['mean_in_batch_accuracy']:.3f} in {summary['seconds']}s"
        )

        checkpoint = config.output_dir / f"epoch-{epoch + 1}"
        encoder.save(checkpoint, tokenizer)
        save_training_state(state_path, epoch + 1, encoder, optimizer, scheduler, history, config)
        on_log(f"wrote {checkpoint}")

    return history


def save_training_state(
    path: Path,
    completed_epochs: int,
    encoder: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    history: TrainingHistory,
    config: TrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "completed_epochs": completed_epochs,
            "model": encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history.as_dict(),
            "config": {**asdict(config), "output_dir": str(config.output_dir)},
        },
        path,
    )


def restore_training_state(
    path: Path,
    encoder: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    history: TrainingHistory,
    device: str,
) -> int:
    state = torch.load(path, map_location=device, weights_only=False)
    encoder.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    history.steps.extend(state["history"]["steps"])
    history.epochs.extend(state["history"]["epochs"])
    return int(state["completed_epochs"])


def write_history(history: TrainingHistory, path: Path, config: TrainingConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        **history.as_dict(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
