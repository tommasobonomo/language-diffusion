import torch
from datasets import Dataset, load_dataset, load_dataset_builder
from lightning import LightningDataModule
from torch.utils import data
from transformers import AutoTokenizer, BatchEncoding


def expand_labels(input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Utility function that expands the labels to match the length of the input_ids.
    Expecting input_ids to be a matrix of size BxS and labels to be a matrix of size BxL,
    where L is the number of labels and S is the number of tokens in the sequence.
    Will return the label matrix BxS.
    """
    # Must use full_like to copy the dtype and device of input_ids
    full_labels = torch.full_like(input_ids, -100)
    full_labels[:, : labels.shape[1]] = labels
    return full_labels


class HFDataset(data.Dataset):
    def __init__(self, dataset: Dataset, tokenizer_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset = dataset

    def __getitem__(self, index: int) -> dict[str, str]:
        sample = self.dataset[index]
        return sample

    def __len__(self) -> int:
        return len(self.dataset)

    def _collate_fn(self, raw_batch: list[dict[str, str]]) -> BatchEncoding:
        articles, highlights = [], []
        for sample in raw_batch:
            articles.append(sample["article"])
            highlights.append(sample["highlights"])

        batch = self.tokenizer(
            text=articles,
            text_target=highlights,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        # batch["labels"] = expand_labels(batch["input_ids"], batch["labels"])
        return batch


class HFDataModule(LightningDataModule):
    hf_cnn_dailymail = "cnn_dailymail"
    version = "3.0.0"

    def __init__(
        self,
        tokenizer_name: str = "facebook/bart-base",
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        self.dataset_builder = load_dataset_builder(self.hf_cnn_dailymail, self.version)
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        self.dataset_builder.download_and_prepare()

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = HFDataset(
                load_dataset(self.hf_cnn_dailymail, self.version, split="train"),  # type: ignore
                self.tokenizer_name,
            )

        if stage == "validate" or stage is None:
            self.val_dataset = HFDataset(
                load_dataset(self.hf_cnn_dailymail, self.version, split="validation"),  # type: ignore
                self.tokenizer_name,
            )
        if stage == "test" or stage is None:
            self.test_dataset = HFDataset(
                load_dataset(self.hf_cnn_dailymail, self.version, split="test"),  # type: ignore
                self.tokenizer_name,
            )

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset._collate_fn,
        )

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset._collate_fn,
        )

    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset._collate_fn,
        )
