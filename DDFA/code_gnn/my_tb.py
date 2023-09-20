from pytorch_lightning.utilities.cli import LOGGER_REGISTRY
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

@LOGGER_REGISTRY
class MyTensorBoardLogger(TensorBoardLogger):
    """Custom subclass of TensorBoardLogger with defaults set."""
    def __init__(self, save_dir=".", default_hp_metric=False, *args, **kwargs):
        super().__init__(save_dir=save_dir, default_hp_metric=default_hp_metric, *args, **kwargs)
