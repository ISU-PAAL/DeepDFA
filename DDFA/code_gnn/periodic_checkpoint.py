import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY


@CALLBACK_REGISTRY
class PeriodicModelCheckpoint(ModelCheckpoint):
    def __init__(self, every: int, **kwargs):
        super().__init__(every_n_epochs=1, **kwargs)
        self.every = every

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ):
        if pl_module.current_epoch % self.every == 0:
            assert self.dirpath is not None
            current = (
                Path(self.dirpath)
                / f"periodical-{pl_module.current_epoch:02d}-{pl_module.global_step:02d}.ckpt"
            )
            trainer.save_checkpoint(current)
