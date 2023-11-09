import pytest

from src.data import HFDataModule


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


def iterate_over_dataloader(dataloader):
    for _ in dataloader:
        pass

    return True


def test_datamodule(datamodule):
    assert iterate_over_dataloader(datamodule.train_dataloader())
    assert iterate_over_dataloader(datamodule.val_dataloader())
    assert iterate_over_dataloader(datamodule.test_dataloader())
