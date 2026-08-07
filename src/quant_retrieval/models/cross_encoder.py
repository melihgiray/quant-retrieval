"""A cross-encoder that scores one question against one answer at a time.

The bi-encoder has to turn every answer into a vector before it ever sees a
question, so the two are compared only after both have been squeezed into 384
numbers. That is what makes it fast enough to search 26,152 documents: the
corpus is embedded once. It is also what limits it, because the model never gets
to look at the question and the answer together.

A cross-encoder gives that up. It concatenates the two into one sequence and runs
attention across both, so a token in the answer can attend to a token in the
question. Much better at judging, and far too slow to search with: scoring the
whole corpus for one query would mean 26,152 forward passes. So it runs last,
over the handful of candidates something cheaper already found.

Initialised from the base pretrained model rather than from the tuned bi-encoder.
The two do different jobs, and starting the judge from the retriever's weights
would make its mistakes correlated with the retriever's, which is the opposite of
what a second stage is for.

It mean pools rather than taking the first token, which is not the usual choice
for a scoring head. The ablation in RESULTS.md is the reason: this base model was
distilled with mean pooling and its CLS token underperforms badly on this corpus,
so the conventional choice would have started from a worse representation.
"""

from __future__ import annotations

from pathlib import Path

from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

from quant_retrieval.models.pooling import POOLING_STRATEGIES, pool


class CrossEncoder(nn.Module):
    """Scores a (question, answer) pair with one number."""

    def __init__(self, model_name: str, pooling: str = "mean") -> None:
        super().__init__()
        if pooling not in POOLING_STRATEGIES:
            raise ValueError(f"unknown pooling {pooling!r}, expected one of {POOLING_STRATEGIES}")
        self.model_name = model_name
        self.pooling = pooling
        self.backbone = AutoModel.from_pretrained(model_name)
        self.score = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, **kwargs) -> Tensor:
        """Return one score per row. Higher means more relevant."""
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled = pool(self.pooling, output.last_hidden_state, attention_mask)
        return self.score(pooled).squeeze(-1)

    def save(self, directory: Path, tokenizer: AutoTokenizer | None = None) -> None:
        """Write the backbone in Hugging Face layout plus the scoring head beside it.

        The head is a single 384 by 1 layer and has nowhere to live in the
        standard layout, so it is saved separately and reloaded by `load`.
        """
        import torch

        directory.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(directory)
        torch.save(
            {"score": self.score.state_dict(), "pooling": self.pooling},
            directory / "scoring_head.pt",
        )
        if tokenizer is not None:
            tokenizer.save_pretrained(directory)

    @classmethod
    def load(cls, directory: Path) -> CrossEncoder:
        import torch

        head = torch.load(directory / "scoring_head.pt", map_location="cpu", weights_only=False)
        model = cls(str(directory), pooling=head.get("pooling", "mean"))
        model.score.load_state_dict(head["score"])
        return model
