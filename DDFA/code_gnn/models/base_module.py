import json
import logging
import random

import dgl
import pytorch_lightning as pl
import torch
import torchmetrics

import torch
from torch import nn
from torch.nn import BCELoss, BCEWithLogitsLoss
import pandas as pd
import numpy as np
from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler

import nni

from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)

logging.getLogger("matplotlib").setLevel(logging.WARNING)


class BaseModule(pl.LightningModule):
    def __init__(
        self, undersample_node_on_loss_factor=None, test_every=False, tune_nni=False, positive_weight=None, profile=False, time=False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.class_threshold = 0.5

        # https://torchmetrics.readthedocs.io/en/stable/pages/classification.html
        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy(),
                torchmetrics.Precision(),
                torchmetrics.Recall(),
                torchmetrics.F1Score(),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        if test_every:
            self.test_every_metrics = metrics.clone(prefix="test_every_")
        else:
            self.test_every_metrics = None
        self.test_metrics = metrics.clone(prefix="test_")
        
        # self.train_metrics_positive = metrics.clone(prefix="train_1_")
        # self.val_metrics_positive = metrics.clone(prefix="val_1_")
        self.test_metrics_positive = metrics.clone(prefix="test_1_")
        
        # self.train_metrics_negative = metrics.clone(prefix="train_0_")
        # self.val_metrics_negative = metrics.clone(prefix="val_0_")
        self.test_metrics_negative = metrics.clone(prefix="test_0_")
        
        self.test_pr_curve = torchmetrics.PrecisionRecallCurve()
        self.test_pr_curve_bin = torchmetrics.BinnedPrecisionRecallCurve(1)
        self.test_preds = torchmetrics.CatMetric()
        self.test_labels = torchmetrics.CatMetric()
        self.test_confmat = torchmetrics.ConfusionMatrix(num_classes=2)
        
        self.label_proportion = nn.ModuleDict({partition + "_label_proportion": torchmetrics.MeanMetric() for partition in ["train", "val", "test"]})
        self.prediction_proportion = nn.ModuleDict({partition + "_prediction_proportion": torchmetrics.MeanMetric() for partition in ["train", "val", "test"]})
        self.label_proportion_cut = nn.ModuleDict({partition + "_label_proportion_cut": torchmetrics.MeanMetric() for partition in ["train", "val", "test"]})
        self.prediction_proportion_cut = nn.ModuleDict({partition + "_prediction_proportion_cut": torchmetrics.MeanMetric() for partition in ["train", "val", "test"]})

        self.train_portion_positive = torchmetrics.MeanMetric()

        if positive_weight is not None:
            positive_weight = torch.tensor([positive_weight])
        self.loss_fn = BCEWithLogitsLoss(pos_weight=positive_weight)

        if profile:
            self.prof = FlopsProfiler(self)
        # if time:
    
    def freeze_graph(self):
        logger.warn("freeze_graph not implemented")

    def get_label(self, batch):
        if self.hparams.label_style == "node":
            label = batch.ndata["_VULN"]
        elif self.hparams.label_style == "graph":
            graphs = dgl.unbatch(batch, batch.batch_num_nodes())
            label = torch.stack([g.ndata["_VULN"].max() for g in graphs])
        elif self.hparams.label_style == "dataflow_solution_out":
            label = batch.ndata["_DF_OUT"]
        elif self.hparams.label_style == "dataflow_solution_in":
            label = batch.ndata["_DF_IN"]
        else:
            raise NotImplementedError(self.hparams.label_style)
        return label.float()

    def resample(self, batch, out, label):
        """Resample logits and labels to balance vuln/nonvuln classes"""
        self.log(
            "meta/train_original_label_proportion",
            torch.mean(label).float().item(),
            on_step=True,
            on_epoch=False,
            batch_size=batch.batch_size,
        )
        self.log(
            "meta/train_original_label_len",
            torch.tensor(label.shape[0]).float(),
            on_step=True,
            on_epoch=False,
            batch_size=batch.batch_size,
        )
        vuln_indices = label.nonzero().squeeze().tolist()
        num_indices_to_sample = round(
            len(vuln_indices) * self.hparams.undersample_node_on_loss_factor
        )
        nonvuln_indices = (label == 0).nonzero().squeeze().tolist()
        nonvuln_indices = random.sample(nonvuln_indices, num_indices_to_sample)
        # TODO: Does this need to be sorted?
        indices = vuln_indices + nonvuln_indices
        out = out[indices]
        label = label[indices]
        self.log(
            "meta/train_resampled_label_proportion",
            torch.mean(label).item(),
            on_step=True,
            on_epoch=False,
            batch_size=batch.batch_size,
        )
        self.log(
            "meta/train_resampled_label_len",
            torch.tensor(label.shape[0]).float(),
            on_step=True,
            on_epoch=False,
            batch_size=batch.batch_size,
        )
        return out, label

    def log_loss(self, name, loss, batch):
        self.log(
            f"{name}_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            batch_size=batch.batch_size,
        )

    def cut_nodef(self, batch, label, out, name):
        # self.log_proportions(label, out, False, name)
        # nz_idx = (batch.ndata["_ABS_DATAFLOW"] != 0) & (~batch.ndata["_SELECTOR"])
        nz_idx = batch.ndata["_ABS_DATAFLOW"].nonzero()
        label = label[nz_idx]
        out = out[nz_idx]
        # self.log_proportions(label, out, True, name)
        return label, out
    
    def log_proportions(self, label, out, cut, name):
        label_metric = self.label_proportion_cut if cut else self.label_proportion
        pred_metric = self.prediction_proportion_cut if cut else self.prediction_proportion
        label_name = name + "_label_proportion"
        if cut:
            label_name += "_cut"
        pred_name = name + "_prediction_proportion"
        if cut:
            pred_name += "_cut"
        label_metric[label_name](label.mean())
        pred_metric[pred_name]((out > 0.5).float().mean())
        self.log(label_name, label_metric[label_name], on_step=False, on_epoch=True)
        self.log(pred_name, pred_metric[pred_name], on_step=False, on_epoch=True)

    def training_step(self, batch_data, batch_idx):
        batch, extrafeats = batch_data
        label = self.get_label(batch)
        out = self.forward(batch, extrafeats)
        if self.hparams.label_style == "dataflow_solution_in":
            label, out = self.cut_nodef(batch, label, out, "train")

        if (
            self.hparams.label_style == "node"
            and self.hparams.undersample_node_on_loss_factor is not None
        ):
            out, label = self.resample(batch, out, label)
        loss = self.loss_fn(out, label)
        self.log_loss("train", loss, batch)

        out = torch.sigmoid(out)
        label = label.int()

        self.log("train_portion_positive", self.train_portion_positive((label == 1).float().mean()), on_step=True, on_epoch=True, batch_size=len(label))
        self.train_metrics.update(out, label)
        # i_pos = torch.nonzero(label == 1)
        # if i_pos.shape[0] > 0:
        #     self.train_metrics_positive.update(out[i_pos], label[i_pos])
        # i_neg = torch.nonzero(label == 0)
        # if i_neg.shape[0] > 0:
        #     self.train_metrics_negative.update(out[i_neg], label[i_neg])
        # assert len(i_pos) + len(i_neg) == len(label)

        return loss

    # def on_after_backward(self):
    #     """https://github.com/Lightning-AI/lightning/issues/2660#issuecomment-699020383"""
    #     if self.global_step % 100 == 0:  # don't make the tf file huge
    #         for name, param in self.named_parameters():
    #             self.logger.experiment.add_histogram(name, param, self.global_step)
    #             if param.requires_grad:
    #                 self.logger.experiment.add_histogram(
    #                     f"{name}_grad", param.grad, self.global_step
    #                 )

    def validation_step(self, batch_data, batch_idx, dataloader_idx=0):
        batch, extrafeats = batch_data
        label = self.get_label(batch)
        out = self.forward(batch, extrafeats)
        if self.hparams.label_style == "dataflow_solution_in":
            label, out = self.cut_nodef(batch, label, out, "val")
        loss = self.loss_fn(out, label)

        out = torch.sigmoid(out)
        label = label.int()

        self.log_loss("val", loss, batch)
        self.val_metrics.update(out, label)
        # i_pos = torch.nonzero(label == 1)
        # if i_pos.shape[0] > 0:
        #     self.val_metrics_positive.update(out[i_pos], label[i_pos])
        # i_neg = torch.nonzero(label == 0)
        # if i_neg.shape[0] > 0:
        #     self.val_metrics_negative.update(out[i_neg], label[i_neg])
        # assert len(i_pos) + len(i_neg) == len(label)
        # if dataloader_idx == 0:  # val set
        #     self.log_loss("val", loss, batch)
        #     self.val_metrics.update(out, label)
        # elif dataloader_idx == 1:  # test set (--test_every)
        #     self.log_loss("test_every", loss, batch)
        #     self.test_every_metrics.update(out, label)

    def test_step(self, batch_data, batch_idx):
        print(batch_idx)
        do_profile = self.hparams.profile and batch_idx > 2
        if do_profile:
            prof = self.prof
        do_time = self.hparams.time and batch_idx > 2
        if do_profile:
            prof.start_profile()
        if do_time:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

        batch, extrafeats = batch_data
        label = self.get_label(batch)
        if do_time:
            start.record()
        out = self.forward(batch, extrafeats)
        if do_time:
            end.record()
        
        profs = []
        if do_profile:
            flops = prof.get_total_flops(as_string=True)
            params = prof.get_total_params(as_string=True)
            macs = prof.get_total_macs(as_string=True)
            prof.print_model_profile(profile_step=batch_idx, output_file=f"./profile.txt")
            prof.end_profile()
            logger.info("step %d: %s flops %s params %s macs", batch_idx, flops, params, macs)
            profs.append({
                "step": batch_idx,
                "flops": flops,
                "params": params,
                "macs": macs,
                "batch_size": len(label),
            })
        if do_time:
            torch.cuda.synchronize()
            tim = start.elapsed_time(end)
            logger.info("step %d: time %f", batch_idx, tim)
            profs.append({
                "step": batch_idx,
                "batch_size": len(label),
                "runtime": tim,
            })
        if do_profile:
            filename = f"profiledata.jsonl"
        elif do_time:
            filename = f"timedata.jsonl"
        else:
            filename = None
        if filename is not None:
            with open(filename, "a") as f:
                f.write(json.dumps(profs[0]))
                f.write("\n")

        if self.hparams.label_style == "dataflow_solution_in":
            label, out = self.cut_nodef(batch, label, out, "test")
            
        if len(out.shape) == 0:
            out = out.unsqueeze(0)
        if len(label.shape) == 0:
            label = label.unsqueeze(0)

        loss = self.loss_fn(out, label)
        self.log_loss("test", loss, batch)
        
        out = torch.sigmoid(out)
        label = label.int()

        self.test_metrics.update(out, label)
        i_pos = torch.nonzero(label == 1)
        # print("positive")
        # print(out[i_pos].shape)
        # print(label[i_pos].shape)
        if i_pos.shape[0] > 0:
            self.test_metrics_positive.update(out[i_pos], label[i_pos])
        i_neg = torch.nonzero(label == 0)
        # print("negative")
        # print(out[i_neg].shape)
        # print(label[i_neg].shape)
        if i_neg.shape[0] > 0:
            self.test_metrics_negative.update(out[i_neg], label[i_neg])
        assert len(i_pos) + len(i_neg) == len(label)

        self.test_preds.update(out)
        self.test_labels.update(label)
        
    def training_epoch_end(self, outputs):
        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)
        # self.log_dict(self.train_metrics_positive.compute(), on_step=False, on_epoch=True)
        # self.log_dict(self.train_metrics_negative.compute(), on_step=False, on_epoch=True)
        self.train_metrics.reset()
        # self.train_metrics_positive.reset()
        # self.train_metrics_negative.reset()
    
    def validation_epoch_end(self, outputs):
        ld = self.val_metrics.compute()
        self.log_dict(ld, on_step=False, on_epoch=True)
        # self.log_dict(self.val_metrics_positive.compute(), on_step=False, on_epoch=True)
        # self.log_dict(self.val_metrics_negative.compute(), on_step=False, on_epoch=True)
        self.val_metrics.reset()
        # self.val_metrics_positive.reset()
        # self.val_metrics_negative.reset()
        # if self.test_every_metrics is not None:
        #     self.log_dict(self.test_every_metrics.compute(), on_step=False, on_epoch=True)
        #     self.test_every_metrics.reset()
        print("intermediate result:", ld)
        # if self.hparams.tune_nni:
        nni.report_intermediate_result(ld["val_F1Score"].cpu().item())
    
    def test_epoch_end(self, outputs):
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)
        self.log_dict(self.test_metrics_positive.compute(), on_step=False, on_epoch=True)
        self.log_dict(self.test_metrics_negative.compute(), on_step=False, on_epoch=True)
        self.test_metrics.reset()
        self.test_metrics_positive.reset()
        self.test_metrics_negative.reset()

        preds, labels = self.test_preds.compute(), self.test_labels.compute().int()

        precision, recall, thresholds = self.test_pr_curve(preds, labels)
        pd.DataFrame({"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": thresholds.tolist() + [1]}).to_csv("pr.csv")
        precision_bin, recall_bin, thresholds_bin = self.test_pr_curve_bin(preds, labels)
        pd.DataFrame({"precision": precision_bin.tolist(), "recall": recall_bin.tolist(), "thresholds": thresholds_bin.tolist() + [1]}).to_csv("pr_binned.csv")

        preds, labels = preds.cpu().numpy(), labels.cpu().numpy()
        preds = preds > 0.5

        print(preds)
        print(labels)

        def get_n_params(model):
            pp=0
            for p in list(model.parameters()):
                nn=1
                for s in list(p.size()):
                    nn = nn*s
                pp += nn
            return pp
        n_params = get_n_params(self)
        print(f"model {n_params} parameters, model architecture {self}")

        print("classification report")
        print(classification_report(labels, preds))
        print("confusion matrix")
        print(confusion_matrix(labels, preds))
