import os
import re
import sys
import dgl
import numpy as np
from pytorch_lightning.utilities.cli import LightningCLI
import tqdm
from code_gnn.models.flow_gnn.ggnn import FlowGNNGGNNModule
from code_gnn.my_tb import MyTensorBoardLogger
from sastvd.linevd.datamodule import BigVulDatasetLineVDDataModule
import sastvd.helpers.datasets as svdds
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import logging
from datetime import datetime
import warnings
import torch as th
import nni
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import SaveConfigCallback
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import os
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger(nni.__name__).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def setup_transient_log():
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(stream_handler)


class QuitEarlyException(Exception):
    pass


def setup_persistent_log():
    logger.setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

    log_filename = "output_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(file_handler)
    logger.info(
        " ".join(a if re.search(r"\s", a) is None else '"' + a + '"' for a in sys.argv)
    )
    logger.info("Logging to %s", log_filename)
    return file_handler, log_filename



class MyLightningCLI(LightningCLI):
    """
    Reference https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html
    """
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--disable-warnings",
            type=bool,
            default=True,
            help="Whether to print warnings",
        )
        parser.add_argument(
            "--analyze_dataset",
            type=bool,
            default=False,
            help="Only analyze the dataset without running the model",
        )
        parser.add_argument(
            "--freeze_graph",
            type=str,
            default=None,
            help="Freeze the weights of the graph learning stage to a ckpt",
        )
        parser.add_argument("--feat_type")
        parser.add_argument("--feat_limitall")
        parser.add_argument("--feat_limitsubkeys")
        parser.link_arguments("data.feat", "model.feat")
        parser.link_arguments("model.label_style", "data.label_style")
        parser.link_arguments("model.concat_all_absdf", "data.concat_all_absdf")
        parser.link_arguments("data.input_dim", "model.input_dim", apply_on="instantiate")
        parser.link_arguments("data.positive_weight", "model.positive_weight", apply_on="instantiate")

    def before_instantiate_classes(self):
        self.file_handler, self.log_filename = setup_persistent_log()

        global_seed = self.config[f"{self.subcommand}.seed_everything"]
        if global_seed is not None:
            logger.info("seeding extra things with %d", global_seed)
            dgl.seed(global_seed)
        if self.config[f"{self.subcommand}.disable_warnings"]:
            warnings.filterwarnings("ignore", category=PossibleUserWarning)
        optimized_params = nni.get_next_parameter()
        if len(optimized_params) > 0:
            logger.info("Got nni parameters: %s", optimized_params)
            for paramname, value in optimized_params.items():
                self.config[f"{self.subcommand}.{paramname}"] = value
                if paramname == "feat_type":
                    self.config[f"{self.subcommand}.data.feat"] += "_" + value + "_all"
                if paramname == "feat_limitall":
                    self.config[f"{self.subcommand}.data.feat"] += "_limitall_" + str(value)
                    self.config[f"{self.subcommand}.data.feat"] += "_limitsubkeys_" + str(value)
            self.config[f"{self.subcommand}.model.feat"] = self.config[f"{self.subcommand}.data.feat"]
        logger.info("Final parameters: %s", self.config)

    def link_log(self):
        if self.trainer.log_dir is not None:
            os.makedirs(self.trainer.log_dir, exist_ok=True)
            dst_filename = os.path.join(self.trainer.log_dir, "output.log")
            if os.path.exists(dst_filename):
                index = 1
                dst_filename = os.path.join(self.trainer.log_dir, f"output_{index}.log")
                while os.path.exists(dst_filename):
                    index += 1
                    dst_filename = os.path.join(self.trainer.log_dir, f"output_{index}.log")
            logger.info("Hard linking %s to %s", self.log_filename, dst_filename)
            os.link(self.log_filename, dst_filename)

    def before_fit(self):
        if self.config[f"{self.subcommand}.freeze_graph"] is not None:
            checkpoint = th.load(self.config[f"{self.subcommand}.freeze_graph"])
            all_state_dict = checkpoint["state_dict"]
            encoder_state_dict = {k: v for k, v in all_state_dict.items() if not k.startswith("output_layer") and not k.startswith("pooling")}
            # TODO: encoder_state_dict = {k: v for k, v in all_state_dict.items() if not k.startswith("output_layer")}
            self.model.load_state_dict(encoder_state_dict, strict=False)
            self.model.freeze_graph_weights()
            print("after load and freeze", self.model, list(self.model.parameters()))
        self.link_log()

    def before_validate(self):
        self.link_log()

    def before_test(self):
        self.link_log()
        if self.config[f"{self.subcommand}.analyze_dataset"]:
            self.subcommand = None
            self.config[f"{self.subcommand}.trainer.limit_test_batches"] = 0
            logger.info("validation coverage: %f", get_coverage(self.datamodule.val))
            logger.info("test coverage: %f", get_coverage(self.datamodule.test))
            logger.info("train coverage: %f", get_coverage(self.datamodule.train))
            self.unlink_log()
            raise QuitEarlyException()

    def unlink_log(self):
        logger.info("done")
        self.file_handler.flush()
        os.unlink(self.log_filename)
        logger.info("unlinked %s", self.log_filename)

    def after_fit(self):
        self.unlink_log()
        
        # if self.config[f"{self.subcommand}.model.tune_nni"] == True:
        for cb in self.trainer.callbacks:
            if isinstance(cb, SaveConfigCallback):
                self.trainer.callbacks.remove(cb)
                break
        model_ckpt_cb = next(c for c in self.trainer.callbacks if isinstance(c, ModelCheckpoint))
        import glob
        ckpts = glob.glob(os.path.join(model_ckpt_cb.dirpath, "performance-*.ckpt"))
        import re
        perfs = [float(re.search(r"val_loss=([0-9\.]+)\.ckpt", t).group(1)) for t in ckpts]
        ckpt_path = ckpts[np.argmin(perfs)]
        logger.info("load ckpt:%s", ckpt_path)
        metrics = self.trainer.validate(self.model, self.datamodule, ckpt_path=ckpt_path)
        logger.info("final val result: %s", metrics)
        nni.report_final_result(metrics[0]["val_F1Score"])

    def after_validate(self):
        self.unlink_log()

    def after_test(self):
        self.unlink_log()

def get_coverage(ds):
    # TODO: abs_df, abs_df_hashes = svdds.abs_dataflow(ds.feat, False, split="fixed", seed=0)
    _, limit_all = svdds.parse_limits(ds.feat)
    print("limit_all", limit_all)

    # feature stats
    num_defs = []
    num_known = []
    num_unknown = []
    num_nodes = []
    num_nodef = []

    # solution stats
    proportion = []
    proportion_nz = []
    
    # general stats
    skipped_feat = skipped_sol = 0

    logger.info("vul distribution:\n%s\n%s", ds.df.value_counts("vul", dropna=False), ds.df.value_counts("vul", normalize=True, dropna=False))

    printed = 0
    for d, ef in tqdm.tqdm(
            ds,
            total=len(ds),
            desc=ds.partition
        ):
        if d is None:
            continue

        if printed < 5:
            logger.debug("%d %s", printed, d)
            logger.debug("%s", d.ndata)
            
        if "_ABS_DATAFLOW" in d.ndata:
            feats = d.ndata["_ABS_DATAFLOW"]
            assert feats.max().item() <= limit_all+2, f"feats should be less than {limit_all} but max was {feats.max().item()}"
            assert len(feats.shape) == 1, f"feats should be |V| but were {feats.shape}"
            assert feats.shape[0] == d.number_of_nodes(), (feats.shape[0], d.number_of_nodes())
            defs = (feats > 0).int().sum().item()
            nodef = (feats == 0).int().sum().item()
            known = ((feats != 1) & (feats > 0)).int().sum().item()
            unknown = (feats == 1).int().sum().item()
            nodes = feats.shape[0]
            num_defs.append(defs)
            num_known.append(known)
            num_unknown.append(unknown)
            num_nodes.append(nodes)
            num_nodef.append(nodef)
            if printed < 5:
                logger.debug("nodes %d", nodes)
                logger.debug("defs %d", defs)
                logger.debug("nodef %d", nodef)
                logger.debug("known %d", known)
                logger.debug("unknown %d", unknown)
        else:
            skipped_feat += 1

        if "_DF_IN" in d.ndata:
            sol = d.ndata["_DF_IN"]
            assert th.all((sol == 0) | (sol == 1))
            assert len(sol.shape) == 1, f"feats should be |V| x |solution.out| but were {sol.shape}"
            assert sol.shape[0] == d.number_of_nodes(), (sol.shape[0], d.number_of_nodes())
            proportion.append(sol.mean().item())

            feats = d.ndata["_ABS_DATAFLOW"]
            nz_idxs = feats.nonzero()
            sol_nz = sol[nz_idxs]
            proportion_nz.append(sol_nz.mean().item())
            if printed < 5:
                logger.debug("%s", nz_idxs)
                logger.debug("%s", sol)
                logger.debug("%s", sol_nz)
                logger.debug("%.2f", sol_nz.mean().item())
        else:
            skipped_sol += 1

        printed += 1

    # compute general stats
    num_nodes = np.array(num_nodes)
    logger.info("partition: %s", ds.partition)
    logger.info("skipped feat: %d, skipped sol: %d", skipped_feat, skipped_sol)
    logger.info("average num nodes: %f", np.average(num_nodes))

    # compute feature stats
    num_nodef = np.array(num_nodef)
    num_defs = np.array(num_defs)
    num_known = np.array(num_known)
    num_unknown = np.array(num_unknown)
    logger.info("number of graphs with features: %d", len(num_defs))
    logger.info("number of graphs without defs: %d", np.sum(num_defs == 0))
    logger.info("number of graphs with at least 1 unknown: %d", sum(num_unknown > 0))
    logger.info("average num nodef: %.2f", np.average(num_nodef))
    logger.info("average num def: %.2f", np.average(num_defs))
    logger.info("average num known: %.2f", np.average(num_known))
    logger.info("average num unknown: %.2f", np.average(num_unknown))
    logger.info("average percentage def: %.2f", np.average(num_defs / num_nodes)*100)
    logger.info("average percentage nodes known (micro): %.2f", np.sum(num_known) / np.sum(num_nodes) * 100)
    logger.info("average percentage nodes unknown (micro): %.2f", np.sum(num_unknown) / np.sum(num_nodes) * 100)
    logger.info("average percentage nodes known (macro): %.2f", np.average(num_known / num_nodes) * 100)
    logger.info("average percentage nodes unknown (macro): %.2f", np.average(num_unknown / num_nodes) * 100)
    logger.info("average percentage def known (micro): %.2f", np.sum(num_known) / np.sum(num_defs) * 100)
    logger.info("average percentage def unknown (micro): %.2f", np.sum(num_unknown) / np.sum(num_defs) * 100)
    nonzero_indices = np.nonzero(num_defs)
    logger.info("average percentage def known (micro) from graphs with defs: %.2f", np.sum(num_known[nonzero_indices]) / np.sum(num_defs[nonzero_indices]) * 100)
    logger.info("average percentage def unknown (micro) from graphs with defs: %.2f", np.sum(num_unknown[nonzero_indices]) / np.sum(num_defs[nonzero_indices]) * 100)
    logger.info("average percentage def known (macro) from graphs with defs: %.2f", np.average(num_known[nonzero_indices] / num_defs[nonzero_indices]) * 100)
    logger.info("average percentage def unknown (macro) from graphs with defs: %.2f", np.average(num_unknown[nonzero_indices] / num_defs[nonzero_indices]) * 100)

    # compute dataflow solution stats
    if len(proportion) > 0:
        proportion = np.array(proportion)
        proportion_nz = np.array(proportion_nz)
        proportion_nz_valid = proportion_nz[~np.isnan(proportion_nz)]
        proportion_nz_na = proportion_nz[np.isnan(proportion_nz)]
        logger.info("average proportion dataflow: %.2f", np.average(proportion))
        logger.info("average proportion definitions dataflow: %.2f", np.average(proportion_nz_valid))
        logger.info("num proportion definitions nan: %d", len(proportion_nz_na))
        logger.info("proportion proportion definitions nan: %.2f", (len(proportion_nz_na)/len(proportion_nz)))
    
    return np.average(num_known[nonzero_indices] / num_defs[nonzero_indices])

if __name__ == "__main__":
    try:
        setup_transient_log()
        MyLightningCLI(FlowGNNGGNNModule, BigVulDatasetLineVDDataModule, parser_kwargs={
            "fit": {"default_config_files": ["configs/config_default.yaml"]},
            "test": {"default_config_files": ["configs/config_default.yaml"]},
        }, save_config_overwrite=True)
    except QuitEarlyException:
        logger.info("Quitting early")
    except:
        logger.error("Error training model", exc_info=True)
        log_filename = None
        root_logger = logging.getLogger()
        print(root_logger.handlers)
        for h in root_logger.handlers:
            if isinstance(h, logging.FileHandler):
                log_filename = h.baseFilename
                break
        logging.shutdown()
        if log_filename is not None:
            shutil.move(log_filename, f"{log_filename}.error")
        raise
