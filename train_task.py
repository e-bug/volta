# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import yaml
import random
import logging
import argparse
from io import open
from tqdm import tqdm
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.distributed as dist
# from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts

from pytorch_transformers.optimization import AdamW, WarmupConstantSchedule, WarmupLinearSchedule

from volta.config import BertConfig, M3PConfig
from volta.optimization import RAdam
from volta.encoders import BertForVLTasks, M3PForVLTasks
from volta.train_utils import freeze_layers, tbLogger, summary_parameters, save, resume
from volta.task_utils import LoadDataset, LoadLoss, ForwardModelsTrain, ForwardModelsVal

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--from_pretrained", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--is_m3p", action='store_true', default=False,
                        help="Use M3P.")
    parser.add_argument("--config_file", default="config/vilbert_base.json", type=str,
                        help="The config file which specified the model details.")
    parser.add_argument("--resume_file", default="", type=str,
                        help="Resume from checkpoint")
    # Output
    parser.add_argument("--output_dir", default="save", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--logdir", default="logs", type=str,
                        help="The logging directory where the training logs will be written.")
    parser.add_argument("--save_name", default="", type=str,
                        help="save name for training.")
    parser.add_argument("--save_best_only", default=False, action="store_true")
    parser.add_argument("--save_every_num_epochs", default=1, type=int)
    # Task
    parser.add_argument("--train_split", default="", type=str)
    parser.add_argument("--val_split", default="", type=str)
    parser.add_argument("--tasks_config_file", default="config_tasks/vilbert_trainval_tasks.yml", type=str,
                        help="The config file which specified the tasks details.")
    parser.add_argument("--task", default="", type=str,
                        help="training task number")
    parser.add_argument("--train_annotations_jsonpath", default="", type=str,
                        help="train_annotations_jsonpath")
    parser.add_argument("--val_annotations_jsonpath", default="", type=str,
                        help="val_annotations_jsonpath")
    parser.add_argument("--train_features_lmdbpath", default="", type=str)
    # Training
    parser.add_argument("--num_epoch", default=None, type=int,
                        help="Max number of training epochs to perform.")
    parser.add_argument("--optim_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", dest="grad_acc_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--drop_last", action="store_true",
                        help="whether to drop last incomplete batch")
    parser.add_argument("--batch_size", default=None, type=int, 
                        help="overwrites the config_tasks batch size")
    parser.add_argument("--eval_batch_size", default=None, type=int,
                        help="overwrites the config_tasks batch size")
    parser.add_argument("--max_val_batches", default=-1, type=int)
    parser.add_argument("--loss", default="", type=str,
                        help="alternative loss name")
    parser.add_argument("--eval_steps", default=sys.maxsize, type=int,
                        help="when to evaluate model")
    parser.add_argument("--cache", default=5000, type=int)
    # Scheduler
    parser.add_argument("--lr", default=None, type=float, 
                        help="overwrites the config_tasks learning rate")
    parser.add_argument("--lr_scheduler", default="warmup_linear", type=str,
                        help="whether use learning rate scheduler.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_steps", default=None, type=float,
                        help="Number of training steps to perform linear learning rate warmup for. "
                             "It overwrites --warmup_proportion.")
    # Seed
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for initialization")
    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of workers in the dataloader.")
    parser.add_argument("--num_val_workers", type=int, default=2)
    parser.add_argument("--in_memory", default=False, type=bool,
                        help="whether use chunck for parallel training.")
    # Optimization
    parser.add_argument("--optim", default="AdamW", type=str,
                        help="what to use for the optimization.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_betas", default=(0.9, 0.999), nargs="+", type=float,
                        help="Betas for Adam optimizer.")
    parser.add_argument("--adam_correct_bias", default=False, action='store_true',
                        help="Correct bias for Adam optimizer.")
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
        torch.distributed.init_process_group(backend="nccl")
    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True
    logger.info(f"device: {device} n_gpu: {n_gpu}, distributed training: {bool(args.local_rank != -1)}")

    # Load config
    if args.is_m3p:
        config = M3PConfig.from_json_file(args.config_file)
    else:
        config = BertConfig.from_json_file(args.config_file)

    # Load task config
    with open(args.tasks_config_file, "r") as f:
        task_cfg = edict(yaml.safe_load(f))
    task_id = args.task.strip()
    task = "TASK" + task_id
    task_name = task_cfg[task]["name"]
    base_lr = args.lr or task_cfg[task]["lr"]
    if task_cfg[task].get("fusion_method", None):
        # VL-BERT pooling for VQA
        config.fusion_method = task_cfg[task]["fusion_method"]

    # Output dirs
    if args.save_name:
        prefix = "-" + args.save_name
    else:
        prefix = ""
    timestamp = (task_name + "_" + args.config_file.split("/")[1].split(".")[0] + prefix)
    save_path = os.path.join(args.output_dir, timestamp)
    if default_gpu:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save all the hidden parameters.
        with open(os.path.join(save_path, "command.txt"), "w") as f:
            print(args, file=f)  # Python 3.x
            print("\n", file=f)
            print(config, file=f)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Dataset
    batch_size, task2num_iters, dset_train, dset_val, dl_train, dl_val = LoadDataset(args, config, task_cfg, args.task)

    # Logging
    logdir = os.path.join(args.logdir, timestamp)
    tb_logger = tbLogger(logdir, save_path, [task_name], [task], task2num_iters, args.grad_acc_steps)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Model
    if "roberta" in config.bert_model:
        config.model = "roberta"
    if args.is_m3p:
        model = M3PForVLTasks.from_pretrained(args.from_pretrained, config=config, task_cfg=task_cfg, task_ids=[task])
    else:
        model = BertForVLTasks.from_pretrained(args.from_pretrained, config=config, task_cfg=task_cfg, task_ids=[task])
    if task_cfg[task].get("embed_clf", None):
        logger.info('Initializing classifier weight for %s from pretrained word embeddings...' % task)
        answers_word_embed = []
        for k, v in model.state_dict().items():
            if 'bert.embeddings.word_embeddings.weight' in k:
                word_embeddings = v.detach().clone()
                break
        for answer, label in sorted(dset_train.ans2label.items()):
            a_tokens = dset_train._tokenizer.tokenize(answer)
            a_ids = dset_train._tokenizer.convert_tokens_to_ids(a_tokens)
            if len(a_ids):
                a_word_embed = (torch.stack([word_embeddings[a_id] for a_id in a_ids], dim=0)).mean(dim=0)
            else:
                a_tokens = dset_train._tokenizer.tokenize("<unk>")
                a_id = dset_train._tokenizer.convert_tokens_to_ids(a_tokens)[0]
                a_word_embed = word_embeddings[a_id]
            answers_word_embed.append(a_word_embed)
        answers_word_embed_tensor = torch.stack(answers_word_embed, dim=0)
        for name, module in model.named_modules():
            if name.endswith('clfs_dict.%s.logit_fc.3' % task):
                module.weight.data = answers_word_embed_tensor.to(device=module.weight.data.device)

    # Optimization details
    freeze_layers(model)
    criterion = LoadLoss(args, task_cfg, args.task)
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if "vil_" in key:
                lr = 1e-4
            else:
                lr = base_lr
            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [{"params": [value], "lr": lr, "weight_decay": 0.0}]
            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [{"params": [value], "lr": lr, "weight_decay": args.weight_decay}]
    if default_gpu:
        print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))
    if args.optim == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=base_lr,
                          eps=args.adam_epsilon,
                          betas=args.adam_betas,
                          correct_bias=args.adam_correct_bias)
    elif args.optim == "RAdam":
        optimizer = RAdam(optimizer_grouped_parameters, lr=base_lr)
    num_train_optim_steps = (task2num_iters[task] * args.optim_train_epochs // args.grad_acc_steps)
    warmup_steps = args.warmup_steps or args.warmup_proportion * num_train_optim_steps
    if args.lr_scheduler == "warmup_linear":
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optim_steps)
    else:
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=warmup_steps)

    # Resume training
    start_iter_id, global_step, start_epoch, tb_logger, max_score = \
        resume(args.resume_file, model, optimizer, scheduler, tb_logger)

    # Move to GPU(s)
    model.to(device)
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
        model = DDP(model, delay_allreduce=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Save starting model
    if start_epoch == 0:
        save(save_path, logger, -1, model, optimizer, scheduler, global_step, tb_logger, default_gpu, max_score)

    # Print summary
    if default_gpu:
        summary_parameters(model, logger)
        print("***** Running training *****")
        print("  Num Iters: ", task2num_iters[task])
        print("  Batch size: ", batch_size)
        print("  Num steps: %d" % num_train_optim_steps)

    # Train
    scores = 0
    max_epoch = args.num_epoch or task_cfg[task]['num_epoch']
    for epoch_id in tqdm(range(start_epoch, max_epoch), desc="Epoch"):
        model.train()
        # from pudb import set_trace; set_trace()
        for step, batch in enumerate(dl_train):
            iter_id = start_iter_id + step // args.grad_acc_steps + (epoch_id * len(dl_train))

            loss, score = ForwardModelsTrain(config, task_cfg, device, task, batch, model, criterion)
            scores += score

            if args.grad_acc_steps > 1:
                loss = loss / args.grad_acc_steps
            loss.backward()

            if (step + 1) % args.grad_acc_steps == 0:
                
                # Clip gradient
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                optimizer.step()

                if global_step < warmup_steps or args.lr_scheduler == "warmup_linear":
                    scheduler.step()

                model.zero_grad()
                global_step += 1

                if default_gpu:
                    tb_logger.step_train(epoch_id, iter_id, float(loss), float(scores/args.grad_acc_steps),
                                         optimizer.param_groups[0]["lr"], task, "train")
                    scores = 0

            if (step % (20 * args.grad_acc_steps) == 0) and step != 0 and default_gpu:
                tb_logger.showLossTrain()

            # Decide whether to evaluate task
            if iter_id != 0 and iter_id % (args.eval_steps - 1) == 0:
                score = evaluate(config, dl_val, task_cfg, device, task, model, criterion, epoch_id, default_gpu, tb_logger)
                if score > max_score:
                    max_score = score
                    save(save_path, logger, iter_id, model, optimizer, scheduler,
                         global_step, tb_logger, default_gpu, max_score, is_best=True)

        torch.cuda.empty_cache()

        score = evaluate(config, dl_val, task_cfg, device, task, model, criterion, epoch_id, default_gpu, tb_logger, args.max_val_batches)
        if score > max_score:
            max_score = score
            save(save_path, logger, epoch_id, model, optimizer, scheduler, global_step, tb_logger, default_gpu, max_score, is_best=True)
        elif (not args.save_best_only) and ((epoch_id+1) % args.save_every_num_epochs == 0):
            save(save_path, logger, epoch_id, model, optimizer, scheduler, global_step, tb_logger, default_gpu, max_score)

    tb_logger.txt_close()
    print("Best Validation score: %.3f " % (max_score * 100.0))


def evaluate(config, dataloader_val, task_cfg, device, task_id, model, criterion, epoch_id, default_gpu, tb_logger, num_batches=-1):
    model.eval()
    for i, batch in enumerate(dataloader_val):
        if i == (num_batches - 1):
            break
        loss, score, batch_size = ForwardModelsVal(config, task_cfg, device, task_id, batch, model, criterion)
        tb_logger.step_val(epoch_id, float(loss), float(score), task_id, batch_size, "val")
        if default_gpu:
            sys.stdout.write("%d/%d\r" % (i, len(dataloader_val)))
            sys.stdout.flush()
    score = tb_logger.showLossVal(task_id)
    model.train()
    return score


if __name__ == "__main__":
    main()
