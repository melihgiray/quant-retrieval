"""Train the cross-encoder that reorders a shortlist.

Same skeleton as the retriever's loop, one difference that matters: this model
sees one question against its own handful of candidates, never against the whole
batch. That is deliberate rather than a simplification. At search time the
reranker only ever gets a shortlist another component produced, so training it
against 300 random documents would be training it for a job it does not do.

The candidates are the hard negatives mined in the previous round, the ones that
did not help the bi-encoder. A reranker is where they belong: it is asked to
separate the right answer from things that already look right, which is exactly
what mining selects for.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from quant_retrieval.models.cross_encoder import CrossEncoder
from quant_retrieval.models.dataset import CrossEncoderCollator, TrainingPair
from quant_retrieval.models.encoder import parameter_groups
from quant_retrieval.models.loss import grouped_cross_entropy, top_one_accuracy
from quant_retrieval.models.schedule import linear_warmup_decay
from quant_retrieval.models.train import (
    STATE_FILE,
    TrainingConfig,
    TrainingHistory,
    restore_training_state,
    save_training_state,
)
from quant_retrieval.runtime import choose_device, set_seed


def build_reranker_dataloader(
    pairs: Sequence[TrainingPair], config: TrainingConfig, tokenizer
) -> DataLoader:
    """Batches are groups of candidates, so batch_size counts questions."""
    if len(pairs) < config.batch_size:
        raise ValueError(f"{len(pairs)} pairs is fewer than one batch of {config.batch_size}")
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        list(pairs),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=CrossEncoderCollator(tokenizer, config.max_length),
        generator=generator,
        num_workers=0,
    )


def train_reranker(
    config: TrainingConfig,
    pairs: Sequence[TrainingPair],
    *,
    resume: bool = False,
    on_log: Callable[[str], None] = print,
) -> TrainingHistory:
    """Fine-tune the cross-encoder, one checkpoint per epoch."""
    if config.negatives_per_query < 1:
        raise ValueError("the reranker needs negatives_per_query of at least 1")

    set_seed(config.seed)
    device = choose_device(config.device)

    if config.limit_pairs is not None:
        pairs = list(pairs)[: config.limit_pairs]

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = CrossEncoder(config.model_name, pooling=config.pooling).to(device)
    if config.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    loader = build_reranker_dataloader(pairs, config, tokenizer)

    total_steps = len(loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    optimizer = AdamW(parameter_groups(model, config.weight_decay), lr=config.learning_rate)
    scheduler = LambdaLR(
        optimizer, lambda step: linear_warmup_decay(step, total_steps, warmup_steps)
    )

    history = TrainingHistory()
    first_epoch = 0
    state_path = config.output_dir / STATE_FILE
    if resume and state_path.exists():
        first_epoch = restore_training_state(
            state_path, model, optimizer, scheduler, history, device
        )
        on_log(f"resumed after epoch {first_epoch}")

    on_log(
        f"{len(pairs)} questions, {config.negatives_per_query + 1} candidates each, "
        f"{len(loader)} steps per epoch, {total_steps} total, device {device}"
    )

    global_step = first_epoch * len(loader)
    for epoch in range(first_epoch, config.epochs):
        model.train()
        epoch_started = time.perf_counter()
        losses: list[float] = []
        accuracies: list[float] = []

        for batch in loader:
            group_size = int(batch.pop("group_size"))
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            flat_scores = model(**batch)
            scores = flat_scores.reshape(-1, group_size)
            loss = grouped_cross_entropy(scores)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            accuracies.append(top_one_accuracy(scores.detach()))
            global_step += 1

            if global_step % config.log_every == 0:
                entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": round(float(loss.item()), 4),
                    "top_one_accuracy": round(accuracies[-1], 4),
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                history.steps.append(entry)
                on_log(
                    f"step {global_step}/{total_steps} loss {entry['loss']:.4f} "
                    f"top1 {entry['top_one_accuracy']:.3f}"
                )

        summary = {
            "epoch": epoch,
            "mean_loss": round(sum(losses) / len(losses), 4),
            "mean_top_one_accuracy": round(sum(accuracies) / len(accuracies), 4),
            "seconds": round(time.perf_counter() - epoch_started, 1),
        }
        history.epochs.append(summary)
        on_log(
            f"epoch {epoch} mean loss {summary['mean_loss']:.4f} "
            f"top1 {summary['mean_top_one_accuracy']:.3f} in {summary['seconds']}s"
        )

        checkpoint = config.output_dir / f"epoch-{epoch + 1}"
        model.save(checkpoint, tokenizer)
        save_training_state(state_path, epoch + 1, model, optimizer, scheduler, history, config)
        on_log(f"wrote {checkpoint}")

    return history
