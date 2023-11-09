import pytest

from src.data import HFDataModule
from src.model.bart import BartBaseline


@pytest.fixture
def datamodule():
    datamodule = HFDataModule(
        tokenizer_name="facebook/bart-base",
        batch_size=32,
        num_workers=4,
    )
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


@pytest.fixture
def bart_model():
    return BartBaseline(
        transformer_name="facebook/bart-base",
    )


def test_bart_single_forward(bart_model, datamodule):
    single_batch = next(iter(datamodule.train_dataloader()))

    assert bart_model._step(single_batch)
