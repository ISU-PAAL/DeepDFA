import sastvd.helpers.datasets as svdds
from sastvd.linevd.dataset import BigVulDatasetLineVD

from torch.utils.data import Subset
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from dgl.dataloading import GraphDataLoader
from torchsampler import ImbalancedDatasetSampler

import logging


logger = logging.getLogger(__name__)


@DATAMODULE_REGISTRY
class BigVulDatasetLineVDDataModule(pl.LightningDataModule):
    """Pytorch Lightning Datamodule for Bigvul."""

    def __init__(
        self,
        feat,
        gtype,
        label_style,
        dsname,
        batch_size=256,
        seed=0,
        sample=-1,
        sample_mode=False,
        split="fixed",
        undersample=None,
        oversample=None,
        train_workers=4,
        val_workers=0,
        test_workers=0,
        concat_all_absdf=False,
        use_weighted_loss=False,
        use_random_weighted_sampler=False,
        train_includes_all=False,
        load_features=True,
    ):
        """Init class from bigvul dataset."""
        super().__init__()
        self.save_hyperparameters()
        dataargs = {
            "sample": sample,
            "sample_mode": sample_mode,
            "gtype": gtype,
            "feat": feat,
            "dsname": dsname,
            "undersample": undersample,
            "oversample": oversample,
            "split": split,
            "seed": seed,
            "label_style": label_style,
            "concat_all_absdf": concat_all_absdf,
            "load_features": load_features,
        }
        self.feat = feat
        self.sample_mode = sample_mode
        self.dsname = dsname

        logger.info("Data args: %s", dataargs)
        if train_includes_all:
            self.train = BigVulDatasetLineVD(partition="all", **dataargs)
        else:
            self.train = BigVulDatasetLineVD(partition="train", **dataargs)
            self.val = BigVulDatasetLineVD(partition="val", **dataargs)
            self.test = BigVulDatasetLineVD(partition="test", **dataargs)

            if "codebert" in dataargs:
                del dataargs["codebert"].tokenizer

            if not sample_mode:
                duped_examples_trainval = set(self.train.df["id"]) & set(self.val.df["id"])
                assert len(duped_examples_trainval) == 0, len(duped_examples_trainval)
                duped_examples_valtest = set(self.val.df["id"]) & set(self.test.df["id"])
                assert len(duped_examples_valtest) == 0, len(duped_examples_valtest)
            logger.debug(f"SPLIT SIZES: {len(self.train)} {len(self.val)} {len(self.test)}")
        self.batch_size = batch_size
        self.train_workers = train_workers
        self.val_workers = val_workers
        self.test_workers = test_workers
        self.use_weighted_loss = use_weighted_loss
        self.use_random_weighted_sampler = use_random_weighted_sampler

    @property
    def input_dim(self):
        if self.feat.startswith("_ABS_DATAFLOW"):
            if "_all" in self.feat:
                _, limit_all = svdds.parse_limits(self.feat)
                return limit_all + 2  # limit to number of hashes plus not-definition and UNKNOWN token
            else:
                raise NotImplementedError("multi-hot")
        else:
            return self.train[0].ndata[self.feat].shape[1]

    @property
    def positive_weight(self):
        if self.use_weighted_loss:
            examples = self.train.df.loc[self.train.idx2id.keys()]
            num_positives = examples["vul"].sum()
            num_negatives = len(examples) - num_positives
            pos_weight = num_negatives / num_positives
            logger.info("Positive weight: %f positives: %d negatives: %d", pos_weight, num_positives, num_negatives)
            return pos_weight
        else:
            return None

    def train_dataloader(self):
        """Return train dataloader."""

        if self.use_random_weighted_sampler:
            sampler = ImbalancedDatasetSampler(self.train)

            return GraphDataLoader(
                self.train,
                # shuffle=True,
                batch_size=self.batch_size,
                num_workers=self.train_workers,
                sampler=sampler,
            )
        else:
            return GraphDataLoader(
                Subset(self.train, self.train.get_epoch_indices()),
                shuffle=True,
                batch_size=self.batch_size,
                num_workers=self.train_workers,
            )

    def val_dataloader(self):
        """Return val dataloader."""
        return GraphDataLoader(self.val, batch_size=self.batch_size, num_workers=self.val_workers)

    def test_dataloader(self):
        """Return test dataloader."""
        return GraphDataLoader(
            self.test,
            batch_size=16,
            num_workers=self.test_workers
        )


def test_dm():
    data = BigVulDatasetLineVDDataModule(
        batch_size=256,
        methodlevel=False,
        gtype="cfg",
        feat="_ABS_DATAFLOW_datatype_all",
        cache_all=False,
        undersample=True,
        filter_cwe=[],
        sample_mode=False,
        use_cache=True,
        train_workers=0,
        split="random",
    )
    print(data)
