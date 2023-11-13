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
        config_filename: str = "config.yaml",
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
        if trainer.logger is not None and stage == "fit":
            wandb_loggers = [
                logger for logger in trainer.loggers if isinstance(logger, WandbLogger)
            ]
            if len(wandb_loggers) > 0:
                wandb_logger = wandb_loggers[0]
                # Write to wandb logging folder if it is a wandb logger
                wandb_logger.experiment.config.update(self.config)
                config_path = (
                    Path("logs")
                    / wandb_logger.experiment.name
                    / (
                        wandb_logger.version
                        if isinstance(wandb_logger.version, str)
                        else str(wandb_logger.version)
                    )
                    / "config.yaml"
                )
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config = self.parser.dump(
                    self.config, skip_none=False, yaml_comments=True
                )
                config_path.write_text(config)


def run():
    cli = LightningCLI(
        subclass_mode_data=True,
        subclass_mode_model=True,
        save_config_callback=CustomSaveConfigCallback,
    )


if __name__ == "__main__":
    run()
