import torch
from lightning import LightningModule
from transformers import BartForConditionalGeneration, BartTokenizer

from src.data import HFDataModule


class BartBaseline(LightningModule):
    def __init__(
        self,
        transformer_name: str = "facebook/bart-base",
        learning_rate: float = 3e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained(transformer_name)
        self.tokenizer = BartTokenizer.from_pretrained(transformer_name)
        self.save_hyperparameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)  # type: ignore

    def _step(self, batch: dict):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels", None)
        outputs = self.forward(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs
