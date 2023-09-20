import logging
import numpy as np
import pandas as pd
import sastvd.helpers.datasets as svdds
import sastvd.helpers.joern as svdj

logger = logging.getLogger(__name__)


def is_valid(_id, hash_index):
    n, e = svdj.get_node_edges(svdds.itempath(_id))
    e = svdj.rdg(e, "cfg")
    n = svdj.drop_lone_nodes(n, e)
    n_hashed = n["id"].apply(lambda nid: hash_index.get((_id, nid), -1) != -1)
    return n_hashed.sum() > 0


class BigVulDataset:
    """Represent BigVul as graph dataset."""

    def __init__(
        self,
        dsname="bigvul",
        partition="train",
        seed=0,
        sample=-1,
        sample_mode=False,
        split="fixed",
        undersample=None,
        oversample=None,
        check_file=True,
        check_valid=True,
        vulonly=False,
    ):
        """Init class."""
        self.partition = partition
        self.undersample = undersample
        self.oversample = oversample

        # load all functions
        df = svdds.ds(dsname, sample=sample_mode)
        if sample != -1:
            df = df.sample(sample, random_state=seed)

        # filter out invalid examples
        df = svdds.ds_filter(
            df,
            dsname,
            check_file=check_file,
            check_valid=check_valid,
            vulonly=vulonly,
            load_code=True,
            sample=sample,
            sample_mode=sample_mode,
        )

        # partition into train/valid/test
        if not sample_mode:
            df = svdds.ds_partition(
                df,
                partition,
                dsname,
                split=split,
                seed=seed,
            )
        logger.info(f"{partition} {len(df)}")
        self.df = df

        # get mapping from index to sample ID
        self.df = self.df.reset_index(drop=True)
        self.idx2id = self.get_idx2id()

        self.rng = np.random.RandomState(seed)

    def get_idx2id(self):
        return dict(zip(self.df.index, self.df.id.values))

    def get_vuln_indices(self, _id):
        """Obtain vulnerable lines from sample ID."""
        df = self.df[self.df.id == _id]
        removed = df.removed.item()
        return dict([(i, 1) for i in removed])
    
    def get_epoch_indices(self):
        """
        Get indices of examples to use for this epoch.
        Undersample graphs to rebalance class distribution if specified.
        """
        index = self.df.index
        if self.undersample is not None or self.oversample is not None:
            logger.info("undersampling: %s oversampling: %s. Resampling from:\n%s\n%s", str(self.undersample), str(self.oversample), self.df.value_counts("vul"), self.df.value_counts("vul", normalize=True))
            vul = self.df[self.df.vul == 1]
            nonvul = self.df[self.df.vul == 0]
            if self.undersample is not None:
                if str(self.undersample).startswith("v"):
                    undersample = float(str(self.undersample)[1:])
                    nonvul = nonvul.sample(int(len(vul)*undersample), replace=False, random_state=self.rng)
                else:
                    nonvul = nonvul.sample(int(len(nonvul)*self.undersample), replace=False, random_state=self.rng)
            if self.oversample is not None:
                vul = vul.sample(int(len(vul)*self.oversample), replace=True, random_state=self.rng)
            undersampled_df = pd.concat([vul, nonvul])
            logger.info("Resampled:\n%s\n%s", undersampled_df.value_counts("vul"), undersampled_df.value_counts("vul", normalize=True))
            index = undersampled_df.index
        return index

    def __getitem__(self, idx):
        """Must override."""
        return self.df.iloc[idx].to_dict()

    def __len__(self):
        """Get length of dataset."""
        return len(self.df)

    def __repr__(self):
        """Override representation."""
        vulnperc = round(len(self.df[self.df.vul == 1]) / len(self), 3)
        return f"BigVulDataset(partition={self.partition}, samples={len(self)}, vulnperc={vulnperc})"
