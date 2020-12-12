# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import json
import random
import logging
import argparse
from io import open

import numpy as np

import torch
import torch.distributed as dist

from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

from volta.config import BertConfig
from volta.encoders import BertForVLPreTraining
from volta.datasets import ConceptCapLoaderTrain, ConceptCapLoaderVal
from volta.train_utils import freeze_layers, tbLogger, summary_parameters, save, resume


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--annotations_path", default="datasets/conceptual_caption/annotations", type=str,
                        help="The corpus annotations directory.")
    parser.add_argument("--features_path", default="datasets/conceptual_caption/imgfeats", type=str,
                        help="The corpus image features directory.")
    # Model
    parser.add_argument("--from_pretrained", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, roberta-base, ...")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, bert-large-uncased, ...")
    parser.add_argument("--config_file", type=str, default="config/vilbert_base.json",
                        help="The config file which specified the model details.")
    parser.add_argument("--resume_file", default="", type=str,
                        help="Resume from checkpoint")
    # Output
    parser.add_argument("--output_dir", default="checkpoints", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--logdir", default="logs", type=str,
                        help="The logging directory where the training logs will be written.")
    # Text
    parser.add_argument("--max_seq_length", default=36, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case", action='store_true', default=True,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    # Training
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", dest="grad_acc_steps", type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    # Scheduler
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_steps", default=None, type=float,
                        help="Number of training steps to perform linear learning rate warmup for. "
                             "It overwrites --warmup_proportion.")
    # Seed
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization")
    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus.")
    parser.add_argument("--num_workers", type=int, default=25,
                        help="Number of workers in the dataloader.")
    parser.add_argument("--distributed", action="store_true",
                        help="whether use chunck for parallel training.")
    # Objective
    parser.add_argument("--objective", default=0, type=int,
                        help="Which objective to use \n"
                             "0: with ITM loss, \n"
                             "1: with ITM loss; for the not aligned pair, no masking objective, \n"
                             "2: without ITM loss, do not sample negative pair.")
    # Optimizer
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_betas", default=(0.9, 0.98), nargs="+", type=float,
                        help="Betas for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay for Adam optimizer.")
    parser.add_argument("--clip_grad_norm", default=0.0, type=float,
                        help="Clip gradients within the specified range.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Devices
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")  # Init distributed backend for sychronizing nodes/GPUs
    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True
    logger.info(f"device: {device} n_gpu: {n_gpu}, distributed training: {bool(args.local_rank != -1)}")

    # Load config
    config = BertConfig.from_json_file(args.config_file)

    # Output dirs
    timestamp = args.config_file.split("/")[1].split(".")[0]
    save_path = os.path.join(args.output_dir, timestamp)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if default_gpu:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save all the hidden parameters.
        with open(os.path.join(save_path, "command.txt"), "w") as f:
            print(args, file=f)  # Python 3.x
            print("\n", file=f)
            print(config, file=f)

    cache = 5000
    args.train_batch_size = args.train_batch_size // args.grad_acc_steps
    if dist.is_available() and args.local_rank != -1:
        num_replicas = dist.get_world_size()
        args.train_batch_size = args.train_batch_size // num_replicas
        args.num_workers = args.num_workers // num_replicas
        cache = cache // num_replicas

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Datasets
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    train_dataset = ConceptCapLoaderTrain(args.annotations_path, args.features_path, tokenizer, args.bert_model,
                                          seq_len=args.max_seq_length, batch_size=args.train_batch_size,
                                          num_workers=args.num_workers, local_rank=args.local_rank,
                                          objective=args.objective, cache=cache,
                                          add_global_imgfeat=config.add_global_imgfeat, num_locs=config.num_locs)
    valid_dataset = ConceptCapLoaderVal(args.annotations_path, args.features_path, tokenizer, args.bert_model,
                                        seq_len=args.max_seq_length, batch_size=args.train_batch_size, num_workers=2,
                                        objective=args.objective, add_global_imgfeat=config.add_global_imgfeat,
                                        num_locs=config.num_locs)

    # Task details
    task_names = ["Conceptual_Caption"]
    task_ids = ["TASK0"]
    task2num_iters = {"TASK0": train_dataset.num_dataset / args.train_batch_size}

    # Logging
    logdir = os.path.join(args.logdir, timestamp)
    if default_gpu:
        tb_logger = tbLogger(logdir, save_path, task_names, task_ids, task2num_iters, args.grad_acc_steps)

    # Model
    if args.from_pretrained:
        type_vocab_size = config.type_vocab_size
        config.type_vocab_size = 2
        model = BertForVLPreTraining.from_pretrained(args.from_pretrained, config=config,
                                                     default_gpu=default_gpu, from_hf=True)
        # Resize type embeddings
        model.bert.embeddings.token_type_embeddings = \
            model._get_resized_embeddings(model.bert.embeddings.token_type_embeddings, type_vocab_size)
        config.type_vocab_size = type_vocab_size
    else:
        model = BertForVLPreTraining(config)

    # Optimization details
    freeze_layers(model)
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    bert_weight_name = json.load(open("config/" + args.from_pretrained + "_weight_name.json", "r"))
    if not args.from_pretrained:
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
    else:
        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if key[12:] in bert_weight_name:
                    lr = args.learning_rate * 0.1
                else:
                    lr = args.learning_rate

                if any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [{"params": [value], "lr": lr, "weight_decay": 0.0}]

                if not any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [{"params": [value], "lr": lr, "weight_decay": args.weight_decay}]
        if default_gpu:
            print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=args.adam_betas)
    num_train_optimization_steps = int(
        train_dataset.num_dataset
        / args.train_batch_size
        / args.grad_acc_steps
    ) * args.num_train_epochs
    warmup_steps = args.warmup_steps or args.warmup_proportion * num_train_optimization_steps
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)

    # Resume training
    start_iter_id, global_step, start_epoch, tb_logger, _ = \
        resume(args.resume_file, model, optimizer, scheduler, tb_logger)

    # Move to GPU(s)
    model.cuda()
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Save starting model
    save(save_path, logger, -1, model, optimizer, scheduler, global_step, tb_logger, default_gpu)

    # Print summary
    if default_gpu:
        summary_parameters(model, logger)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_dataset.num_dataset)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

    # Train
    for epoch_id in range(start_epoch, int(args.num_train_epochs)):
        model.train()
        for step, batch in enumerate(train_dataset):
            iter_id = start_iter_id + step + (epoch_id * len(train_dataset))
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch[:-1])

            input_ids, input_mask, segment_ids, lm_label_ids, is_match, \
            image_feat, image_loc, image_cls, obj_labels, obj_confs, \
            attr_labels, attr_confs, image_attrs, image_label, image_mask = batch

            if args.objective == 1:
                # Ignore labels (setting them to -1) for mismatched caption-image pairs
                image_label = image_label * (is_match == 0).long().unsqueeze(1)
                image_label[image_label == 0] = -1
                lm_label_ids = lm_label_ids * (is_match == 0).long().unsqueeze(1)
                lm_label_ids[lm_label_ids == 0] = -1

            masked_loss_t, masked_loss_v, pair_match_loss = model(input_ids, image_feat, image_loc, segment_ids,
                                                                  input_mask, image_mask, lm_label_ids, image_label,
                                                                  image_cls, obj_labels, obj_confs, attr_labels,
                                                                  attr_confs, image_attrs, is_match)

            if args.objective == 2:
                pair_match_loss = pair_match_loss * 0

            loss = masked_loss_t + masked_loss_v + pair_match_loss
            if n_gpu > 1:
                loss = loss.mean()
                masked_loss_t = masked_loss_t.mean()
                masked_loss_v = masked_loss_v.mean()
                pair_match_loss = pair_match_loss.mean()

            if args.grad_acc_steps > 1:
                loss = loss / args.grad_acc_steps
            loss.backward()

            if (step + 1) % args.grad_acc_steps == 0:
                # Clip gradient
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if default_gpu:
                    tb_logger.step_train_CC(epoch_id, iter_id,
                                            float(masked_loss_t), float(masked_loss_v), float(pair_match_loss),
                                            optimizer.param_groups[0]["lr"], "TASK0", "train")

            if (step % (20 * args.grad_acc_steps) == 0) and step != 0 and default_gpu:
                tb_logger.showLossTrainCC()

        # Do the evaluation
        torch.set_grad_enabled(False)
        numBatches = len(valid_dataset)
        model.eval()
        for step, batch in enumerate(valid_dataset):
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch[:-1])

            input_ids, input_mask, segment_ids, lm_label_ids, is_match, \
            image_feat, image_loc, image_cls, obj_labels, obj_confs, \
            attr_labels, attr_confs, image_attrs, image_label, image_mask = batch

            batch_size = input_ids.size(0)
            masked_loss_t, masked_loss_v, pair_match_loss = model(input_ids, image_feat, image_loc, segment_ids,
                                                                  input_mask, image_mask, lm_label_ids, image_label,
                                                                  image_cls, obj_labels, obj_confs, attr_labels,
                                                                  attr_confs, image_attrs, is_match)

            loss = masked_loss_t + masked_loss_v + pair_match_loss
            if n_gpu > 1:
                loss = loss.mean()
                masked_loss_t = masked_loss_t.mean()
                masked_loss_v = masked_loss_v.mean()
                pair_match_loss = pair_match_loss.mean()

            if default_gpu:
                tb_logger.step_val_CC(epoch_id, float(masked_loss_t), float(masked_loss_v), float(pair_match_loss),
                                      "TASK0", batch_size, "val")
                sys.stdout.write("%d / %d \r" % (step, numBatches))
                sys.stdout.flush()

        torch.set_grad_enabled(True)
        save(save_path, logger, epoch_id, model, optimizer, scheduler, global_step, tb_logger, default_gpu)

    if default_gpu:
        tb_logger.txt_close()


if __name__ == "__main__":
    main()
