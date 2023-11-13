from pathlib import Path
from typing import Any

from jsonargparse import Namespace
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import (
    LightningArgumentParser,
    LightningCLI,
    SaveConfigCallback,
)
from lightning.pytorch.loggers import WandbLogger


class CustomSaveConfigCallback(SaveConfigCallback):
    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace | Any,
        config_filename: str = "cli_config.yaml",
        overwrite: bool = False,
        multifile: bool = False,
        save_to_log_dir: bool = False,
    ) -> None:
        super().__init__(
            parser, config, config_filename, overwrite, multifile, save_to_log_dir
        )

    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        if len(trainer.loggers) > 0 and stage == "fit":
            wandb_loggers = [
                logger for logger in trainer.loggers if isinstance(logger, WandbLogger)
            ]
            if len(wandb_loggers) > 0:
                wandb_logger = wandb_loggers[0]
                config_path = Path(wandb_logger.experiment.dir) / self.config_filename
                config_path.write_text(
                    self.parser.dump(self.config, skip_none=False, yaml_comments=True)
                )


def run():
    cli = LightningCLI(
        subclass_mode_data=True,
        subclass_mode_model=True,
        save_config_callback=CustomSaveConfigCallback,
    )


if __name__ == "__main__":
    run()
