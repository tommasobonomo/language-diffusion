import evaluate
import torch
from lightning import LightningModule
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    BatchEncoding,
    GenerationConfig,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.optimization import get_linear_schedule_with_warmup


class BartBaseline(LightningModule):
    def __init__(
        self,
        transformer_name: str = "facebook/bart-base",
        learning_rate: float = 5e-5,
        weight_decay: float = 0.0,
        exclude_params: list[str] = ["bias", "layer_norm.weight"],
        generation_config: GenerationConfig | None = None,
        num_training_steps: int = 100_000,
    ):
        super().__init__()
        # Initialise model and tokenizer (needed for metrics)
        self.model = BartForConditionalGeneration.from_pretrained(transformer_name)
        self.tokenizer = BartTokenizer.from_pretrained(transformer_name)

        # Save hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.exclude_params = exclude_params
        self.num_training_steps = num_training_steps

        # Initialise metrics module
        self.rouge_metric = evaluate.load("rouge")

        # Save generation config, either default model config or custom config
        if generation_config is None:
            self.generation_config = self.model.generation_config  # type: ignore
        else:
            self.generation_config = generation_config

    def configure_optimizers(self):
        # Optimizer
        grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in self.exclude_params)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in self.exclude_params)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.RAdam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        # Linear schedule with no warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=100_000
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }

    def forward(
        self,
        batch: BatchEncoding,
    ) -> torch.Tensor:
        # Decoding, done following config given in initialisation
        return self.model.generate(  # type: ignore
            batch.input_ids,
            attention_mask=batch.attention_mask,
            generation_config=self.generation_config,
        )

    def _step(self, batch: BatchEncoding) -> Seq2SeqLMOutput:
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        labels = batch.get("labels", None)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)  # type: ignore
        return outputs

    def _predict(self, batch: BatchEncoding) -> torch.Tensor:
        # Greedy decoding, i.e. beam search with num_beams=1
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        return self.model.generate(  # type: ignore
            input_ids,
            attention_mask=attention_mask,
            num_beams=1,
            early_stopping=False,
            do_sample=False,
            max_new_tokens=100,
        )

    def training_step(self, batch: BatchEncoding) -> torch.Tensor:
        # Will include loss if labels are provided in batch, so need to check for that
        output = self._step(batch)
        if batch.get("labels", None) is None or output.loss is None:
            raise RuntimeError("Target sentence not provided in training batch")
        loss = output.loss
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: BatchEncoding) -> torch.Tensor:
        # Labels are required also in the validation step, both for loss and metrics computation
        output = self._step(batch)
        if batch.get("labels", None) is None or output.loss is None:
            raise RuntimeError("Target sentence not provided in validation batch")
        loss = output.loss
        self.log("val/loss", loss, prog_bar=True)
        predictions = self._predict(batch)
        metrics = self.compute_metrics(predictions, batch.labels)
        self.log_dict(metrics, prog_bar=False)
        return loss

    def test_step(self, batch: BatchEncoding) -> torch.Tensor:
        # Labels are required also in the test step, both for loss and metrics computation
        output = self._step(batch)
        if batch.get("labels", None) is None or output.loss is None:
            raise RuntimeError("Target sentence not provided in test batch")
        loss = output.loss
        self.log("test/loss", loss, prog_bar=True)
        predictions = self._predict(batch)
        metrics = self.compute_metrics(predictions, batch.labels)
        self.log_dict(metrics, prog_bar=False)
        return loss

    def predict_step(self, batch: BatchEncoding) -> torch.Tensor:
        # Labels are not required in the predict step
        predictions = self._predict(batch)
        return predictions

    def compute_metrics(
        self, predictions: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        str_predictions = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        str_targets = self.tokenizer.batch_decode(
            target, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        rouge = self.rouge_metric.compute(
            predictions=str_predictions, references=str_targets
        )
        return (
            {f"val/{rouge_metric}": value for rouge_metric, value in rouge.items()}
            if rouge is not None
            else {}
        )
