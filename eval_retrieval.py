# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import yaml
import random
import logging
import argparse
from io import open
from tqdm import tqdm
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist

from volta.config import BertConfig
from volta.encoders import BertForVLTasks, BertForVLPreTraining
from volta.task_utils import LoadDatasetEval, LoadLoss, ForwardModelsTrain, ForwardModelsVal


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
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--config_file", default="config/bert_config.json", type=str,
                        help="The config file which specified the model details.")
    # Output
    parser.add_argument("--output_dir", default="results", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--save_name", default="", type=str,
                        help="save name for training.")
    # Task
    parser.add_argument("--tasks_config_file", default="config_tasks/vilbert_trainval_tasks.yml", type=str,
                        help="The config file which specified the tasks details.")
    parser.add_argument("--task", default="", type=str,
                        help="training task number")
    # Text
    parser.add_argument("--do_lower_case", default=True, type=bool,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    # Evaluation
    parser.add_argument("--split", default="", type=str,
                        help="which split to use.")
    parser.add_argument("--zero_shot", action="store_true",
                        help="Zero-shot evaluation.")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="batch size.")
    parser.add_argument("--drop_last", action="store_true",
                        help="whether to drop last incomplete batch")
    # Seed
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of workers in the dataloader.")
    parser.add_argument("--in_memory", default=False, type=bool,
                        help="whether use chunck for parallel training.")
    parser.add_argument("--use_chunk", default=0, type=float,
                        help="whether use chunck for parallel training.")

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
    config = BertConfig.from_json_file(args.config_file)

    # Load task config
    with open(args.tasks_config_file, "r") as f:
        task_cfg = edict(yaml.safe_load(f))
    task_id = args.task.strip()
    task = "TASK" + task_id

    # Output dirs
    if "/" in args.from_pretrained:
        timeStamp = args.from_pretrained.split("/")[1]
    else:
        timeStamp = args.from_pretrained
    savePath = os.path.join(args.output_dir, timeStamp)
    if default_gpu and not os.path.exists(savePath):
        os.makedirs(savePath)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Dataset
    batch_size, task2num_iters, dset_val, dl_val = LoadDatasetEval(args, config, task_cfg, args.task)

    # Model
    if args.zero_shot:
        config.visual_target_weights = {}
        model = BertForVLPreTraining.from_pretrained(args.from_pretrained, config=config)
    else:
        model = BertForVLTasks.from_pretrained(args.from_pretrained, config=config, task_cfg=task_cfg, task_ids=[task])

    # Move to GPU(s)
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, deay_allreduce=True)
    elif n_gpu > 1:
        model = nn.DataParallel(model)
        raise ValueError("Please run with a single GPU")

    # Print summary
    if default_gpu:
        print("***** Running evaluation *****")
        print("  Num Iters: ", task2num_iters)
        print("  Batch size: ", batch_size)

    # Evaluate
    model.eval()
    results = []
    others = []
    score_matrix = np.zeros((5000, 1000))
    target_matrix = np.zeros((5000, 1000))
    rank_matrix = np.ones(5000) * 1000
    count = 0
    for i, batch in tqdm(enumerate(dl_val), total=task2num_iters[task]):
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
        features, spatials, image_mask, question, input_mask, segment_ids, target, caption_idx, image_idx = batch

        features = features.squeeze(0)
        spatials = spatials.squeeze(0)
        image_mask = image_mask.squeeze(0)
        question = question.repeat(features.size(0), 1)
        segment_ids = segment_ids.repeat(features.size(0), 1)
        input_mask = input_mask.repeat(features.size(0), 1)

        with torch.no_grad():
            if args.zero_shot:
                _, _, vil_logit, _, _ = model(question, features, spatials, segment_ids, input_mask, image_mask)

                score_matrix[
                    caption_idx, image_idx * 500: (image_idx + 1) * 500
                ] = (torch.softmax(vil_logit, dim=1)[:, 0].view(-1).cpu().numpy())
                target_matrix[
                    caption_idx, image_idx * 500: (image_idx + 1) * 500
                ] = (target.view(-1).float().cpu().numpy())

            else:
                vil_logit, _, _, _ = model(question, features, spatials, task, segment_ids, input_mask, image_mask)

                score_matrix[
                    caption_idx, image_idx * 500: (image_idx + 1) * 500
                ] = (vil_logit.view(-1).cpu().numpy())
                target_matrix[
                    caption_idx, image_idx * 500: (image_idx + 1) * 500
                ] = (target.view(-1).float().cpu().numpy())

            if image_idx.item() == 1:
                rank = np.where(
                    (
                        np.argsort(-score_matrix[caption_idx])
                        == np.where(target_matrix[caption_idx] == 1)[0][0]
                    )
                    == 1
                )[0][0]
                rank_matrix[caption_idx] = rank

                rank_matrix_tmp = rank_matrix[: caption_idx + 1]
                r1 = 100.0 * np.sum(rank_matrix_tmp < 1) / len(rank_matrix_tmp)
                r5 = 100.0 * np.sum(rank_matrix_tmp < 5) / len(rank_matrix_tmp)
                r10 = 100.0 * np.sum(rank_matrix_tmp < 10) / len(rank_matrix_tmp)

                medr = np.floor(np.median(rank_matrix_tmp) + 1)
                meanr = np.mean(rank_matrix_tmp) + 1
                print(
                    "%d Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
                    % (count, r1, r5, r10, medr, meanr)
                )

                results.append(np.argsort(-score_matrix[caption_idx]).tolist()[:20])
        count += 1

    r1 = 100.0 * np.sum(rank_matrix < 1) / len(rank_matrix)
    r5 = 100.0 * np.sum(rank_matrix < 5) / len(rank_matrix)
    r10 = 100.0 * np.sum(rank_matrix < 10) / len(rank_matrix)

    medr = np.floor(np.median(rank_matrix) + 1)
    meanr = np.mean(rank_matrix) + 1

    print("************************************************")
    print("****************Image Retrieval*****************")
    print("************************************************")
    print(
        "Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
        % (r1, r5, r10, medr, meanr)
    )
    print("************************************************")

    if args.split:
        json_path = os.path.join(savePath, args.split)
    else:
        json_path = os.path.join(savePath, task_cfg[task_id]["val_split"])
    json.dump(results, open(json_path + "_result.json", "w"))
    json.dump(others, open(json_path + "_others.json", "w"))

    # Text Retrieval
    rank_matrix = np.zeros(1000)
    for image_idx in range(1000):
        ranks = []
        tgt_captions = np.where(target_matrix[:, image_idx] == 1)[0]
        sorted_scores = np.argsort(-score_matrix[:, image_idx])
        for tgt_caption in tgt_captions:
            ranks.append(np.where((sorted_scores == tgt_caption) == 1)[0][0])
        rank_matrix[image_idx] = min(ranks)

    r1 = 100.0 * np.sum(rank_matrix < 1) / len(rank_matrix)
    r5 = 100.0 * np.sum(rank_matrix < 5) / len(rank_matrix)
    r10 = 100.0 * np.sum(rank_matrix < 10) / len(rank_matrix)

    medr = np.floor(np.median(rank_matrix) + 1)
    meanr = np.mean(rank_matrix) + 1

    print("************************************************")
    print("****************Text Retrieval******************")
    print("************************************************")
    print(
        "Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
        % (r1, r5, r10, medr, meanr)
    )
    print("************************************************")


if __name__ == "__main__":
    main()
