# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import logging
import argparse
import math
import numpy as np
from io import open
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
import multiprocessing
import time

from models import DefectModel
from configs import add_args, set_seed
from utils import get_filenames, get_elapse_time, load_and_cache_defect_data
from models import get_model_size

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}

cpu_cont = multiprocessing.cpu_count()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(args, model, eval_examples, eval_data, write_to_pred=False, flowgnn_dataset=None):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Num batches = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    num_missing = 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating"):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        index = batch[2].to(args.device)
        if flowgnn_dataset is None:
            graphs=None
        else:
            graphs, keep_idx = flowgnn_dataset.get_indices(index)
            num_missing_batch = len(index) - len(keep_idx)
            num_missing += num_missing_batch
            inputs = inputs[keep_idx]
            label = label[keep_idx]
            index = index[keep_idx]
            if num_missing_batch > 0:
                logger.info("%d examples missing in batch", num_missing_batch)
            if graphs is None:
                logger.info("skipping batch of %d items, graphs is None for indices: %s", len(index), index)
                continue
        with torch.no_grad():
            lm_loss, logit = model(inputs, label, graphs=graphs)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logger.info("%d examples missing", num_missing)
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits[:, 1] > 0.5
    eval_acc = accuracy_score(labels, preds)
    eval_precision = precision_score(labels, preds)
    eval_recall = recall_score(labels, preds)
    eval_f1 = f1_score(labels, preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
        "eval_precision": round(eval_precision, 4),
        "eval_recall": round(eval_recall, 4),
        "eval_f1": round(eval_f1, 4),
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    if write_to_pred:
        with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
            for example, pred in zip(eval_examples, preds):
                if pred:
                    f.write(str(example.idx) + '\t1\n')
                else:
                    f.write(str(example.idx) + '\t0\n')

    return result


def main():
    parser = argparse.ArgumentParser()
    t0 = time.time()
    args = add_args(parser)
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_cont)
    args.device = device
    set_seed(args)

    # Build model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)

    ### ON MY FLOWGNN SHID
    if args.flowgnn_data:
        logger.info("ACTIVATING FLOWGNN DATA")
        import sys
        sys.path.append("/home/<ANON>/code/ddfa/code_gnn/linevd-code_gnn/linevd")
        from code_gnn.models.flow_gnn.ggnn import FlowGNNGGNNModule
        from sastvd.linevd import BigVulDatasetLineVDDataModule
        # load graphs
        feat = "_ABS_DATAFLOW_datatype_all_limitall_1000_limitsubkeys_1000"
        gtype = "cfg"
        label_style = "graph"
        dsname = "bigvul"
        # concat_all_absdf = False
        node_type_feat = None
        concat_all_absdf = True
        # node_type_feat = "aftergraph"
        flowgnn_datamodule = BigVulDatasetLineVDDataModule(
            feat,
            gtype,
            label_style,
            dsname,
            undersample=None,
            oversample=None,
            filter_cwe="",
            sample=-1,
            sample_mode=False,
            train_workers=1,
            val_workers=0,
            test_workers=0,
            split="fixed",
            batch_size=256,
            # nsampling=False,
            # nsampling_hops=1,
            seed=args.seed,
            # test_every=False,
            # dataflow_defined_only=False,
            # codebert_feat=None,
            # doc2vec_feat=None,
            # glove_feat=None,
            node_type_feat=node_type_feat,
            concat_all_absdf=concat_all_absdf,
            # use_weighted_loss=False,
            # use_random_weighted_sampler=False,
            train_includes_all=True,
            load_features=True,
        )
        flowgnn_dataset = flowgnn_datamodule.train
        logger.info("FlowGNN dataset: %s\n%s", flowgnn_datamodule.train.df, flowgnn_datamodule.train.idx2id)
    else:
        flowgnn_dataset = None

    # load model
    if args.flowgnn_model:
        import sys
        sys.path.append("/home/<ANON>/code/ddfa/code_gnn/linevd-code_gnn/linevd")
        from code_gnn.models.flow_gnn.ggnn import FlowGNNGGNNModule
        from sastvd.linevd import BigVulDatasetLineVDDataModule
        logger.info("ACTIVATING FLOWGNN MODEL")
        input_dim = flowgnn_datamodule.input_dim
        hidden_dim = 32
        n_steps = 5
        num_output_layers = 3
        flowgnn_model = FlowGNNGGNNModule(
            feat,
            input_dim,
            hidden_dim,
            n_steps,
            num_output_layers,
            label_style=label_style,
            # freeze_graph=False,
            # append_dataflow="before_graph",
            # codebert_feat=None,
            # doc2vec_feat=None,
            # glove_feat=None,
            num_node_types=flowgnn_datamodule.num_node_types,
            node_type_feat=node_type_feat,
            # just_codebert=False,
            concat_all_absdf=concat_all_absdf,
            # undersample_node_on_loss_factor=None,
            # test_every=False,
            # tune_nni=False,
            # positive_weight=None,
            encoder_mode=True,
        )
        logger.info("FlowGNN output dim: %d", flowgnn_model.out_dim)
    else:
        flowgnn_model = None


    model = DefectModel(model, config, tokenizer, args, flowgnn_model=flowgnn_model)
    logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)

    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)

    pool = multiprocessing.Pool(cpu_cont)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        if args.n_gpu > 1:
            # multi-gpu training
            model = torch.nn.DataParallel(model)
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_examples, train_data = load_and_cache_defect_data(args, args.train_filename, pool, tokenizer, 'train',
                                                                is_sample=False)
        # train_data = TensorDataset(train_data[:10])
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        save_steps = max(len(train_dataloader), 1)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        if args.warmup_steps < 1:
            warmup_steps = num_train_optimization_steps * args.warmup_steps
        else:
            warmup_steps = int(args.warmup_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        global_step, best_f1 = 0, 0
        not_f1_inc_cnt = 0
        is_early_stop = False
        num_missing = 0
        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(device) for t in batch)
                source_ids, labels, index = batch
                if flowgnn_dataset is None:
                    graphs=None
                else:
                    graphs, keep_idx = flowgnn_dataset.get_indices(index)
                    num_missing_batch = len(index) - len(keep_idx)
                    num_missing += num_missing_batch
                    source_ids = source_ids[keep_idx]
                    labels = labels[keep_idx]
                    index = index[keep_idx]
                    if num_missing_batch > 0:
                        logger.info("%d examples missing in batch", num_missing_batch)
                    if graphs is None:
                        logger.info("skipping batch of %d items, graphs is None for indices: %s", len(index), index)
                        continue

                # TODO: input graphs
                loss, logits = model(source_ids, labels, graphs=graphs)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / nb_tr_steps, 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

                if (step + 1) % save_steps == 0 and args.do_eval:
                    logger.info("***** CUDA.empty_cache() *****")
                    torch.cuda.empty_cache()

                    eval_examples, eval_data = load_and_cache_defect_data(args, args.dev_filename, pool, tokenizer,
                                                                          'valid', is_sample=False)
                    # eval_data = TensorDataset(eval_data[:10])

                    result = evaluate(args, model, eval_examples, eval_data, flowgnn_dataset=flowgnn_dataset)
                    eval_f1 = result['eval_f1']
                    eval_acc = result['eval_acc']

                    if args.data_num == -1:
                        tb_writer.add_scalar('dev_acc', round(eval_acc, 4), cur_epoch)
                        tb_writer.add_scalar('dev_f1', round(eval_f1, 4), cur_epoch)

                    # save last checkpoint
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)

                    if True or args.data_num == -1 and args.save_last_checkpoints:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the last model into %s", output_model_file)

                    if eval_f1 > best_f1:
                        not_f1_inc_cnt = 0
                        logger.info("  Best f1: %s", round(eval_f1, 4))
                        logger.info("  " + "*" * 20)
                        fa.write("[%d] Best acc changed into %.4f\n" % (cur_epoch, round(eval_f1, 4)))
                        best_f1 = eval_f1
                        # Save best checkpoint for best ppl
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or True:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best ppl model into %s", output_model_file)
                    else:
                        not_f1_inc_cnt += 1
                        logger.info("acc does not increase for %d epochs", not_f1_inc_cnt)
                        if not_f1_inc_cnt > args.patience:
                            logger.info("Early stop as f1 do not increase for %d times", not_f1_inc_cnt)
                            fa.write("[%d] Early stop as not_f1_inc_cnt=%d\n" % (cur_epoch, not_f1_inc_cnt))
                            is_early_stop = True
                            break

                model.train()
            if is_early_stop:
                break

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()
        logger.info("%d items missing", num_missing)

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        for criteria in ['best-acc']:
            file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            logger.info("Reload model from {}".format(file))
            model.load_state_dict(torch.load(file))

            if args.n_gpu > 1:
                # multi-gpu training
                model = torch.nn.DataParallel(model)

            eval_examples, eval_data = load_and_cache_defect_data(args, args.test_filename, pool, tokenizer, 'test',
                                                                  False)

            result = evaluate(args, model, eval_examples, eval_data, write_to_pred=True, flowgnn_dataset=flowgnn_dataset)
            logger.info("  test_acc=%.4f", result['eval_acc'])
            logger.info("  test_f1=%.4f", result['eval_f1'])
            logger.info("  " + "*" * 20)

            fa.write("[%s] test-acc: %.4f\n" % (criteria, result['eval_acc']))
            fa.write("[%s] test-f1: %.4f\n" % (criteria, result['eval_f1']))
            if args.res_fn:
                with open(args.res_fn, 'a+') as f:
                    f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                    f.write("[%s] acc: %.4f\n\n" % (
                        criteria, result['eval_acc']))
    fa.close()


if __name__ == "__main__":
    main()
