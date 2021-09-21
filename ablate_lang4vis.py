# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import json
import yaml
import random
import logging
import argparse
from io import open
from tqdm import tqdm
import _pickle as cPickle
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from pytorch_transformers.tokenization_bert import BertTokenizer

from volta.config import BertConfig
from volta.encoders import BertForVLPreTraining
from volta.datasets import FlickrLang4VisDataset
from volta.losses import pre_vis_criterions
from volta.datasets._all_image_features_reader import ImageFeaturesH5Reader


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
    parser.add_argument("--dump_results", default=False, action="store_true",
                        help="Whether to save predictions onto disk")
    # Task
    parser.add_argument("--tasks_config_file", default="config_tasks/vilbert_trainval_tasks.yml", type=str,
                        help="The config file which specified the tasks details.")
    parser.add_argument("--task", default="", type=str,
                        help="training task number")
    parser.add_argument("--masking", default=None, type=str, choices=["all", "phrase", "none"],
                        help="Text tokens to mask")
    parser.add_argument("--overlap_threshold", default=0.5, type=float,
                        help="Threshold for image regions to mask")
    # Text
    parser.add_argument("--do_lower_case", default=True, type=bool,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    # Evaluation
    parser.add_argument("--split", default="", type=str,
                        help="which split to use.")
    parser.add_argument("--batch_size", default=30, type=int,
                        help="batch size.")
    parser.add_argument("--drop_last", action="store_true",
                        help="whether to drop last incomplete batch")
    # Seed
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--num_workers", type=int, default=0,
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
    task_name = task_cfg[task]["name"]
    if task_cfg[task].get("fusion_method", None):
        # VL-BERT pooling for VQA
        config.fusion_method = task_cfg[task]["fusion_method"]

    # Output dirs
    savePath = args.output_dir
    if default_gpu and not os.path.exists(savePath):
        os.makedirs(savePath)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Dataset
    feats_h5path = task_cfg[task]["features_h5path1"]
    features_reader = ImageFeaturesH5Reader(feats_h5path, config, args.in_memory)
    batch_size = task_cfg[task]["batch_size"]
    num_workers = args.num_workers
    if args.local_rank != -1:
        batch_size = int(batch_size / dist.get_world_size())
        num_workers = int(num_workers / dist.get_world_size())
    logger.info("Loading %s Dataset with batch size %d" % (task_name, batch_size))
    eval_split = args.split or task_cfg[task]["val_split"]
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    dset = FlickrLang4VisDataset(
        task, task_cfg[task]["dataroot"], args.masking, eval_split, features_reader, None,
        tokenizer, args.bert_model, max_seq_length=task_cfg[task]["max_seq_length"],
        max_region_num=task_cfg[task]["max_region_num"], num_locs=config.num_locs,
        threshold=args.overlap_threshold, add_global_imgfeat=config.add_global_imgfeat
    )
    dl = DataLoader(dset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Model
    model = BertForVLPreTraining.from_pretrained(args.from_pretrained, config=config)

    # Move to GPU(s)
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, delay_allreduce=True)
    elif n_gpu > 1:
        model = nn.DataParallel(model)

    # Print summary
    if default_gpu:
        print("***** Running evaluation *****")
        print("  Num Iters: ", len(dl))
        print("  Batch size: ", batch_size)
        
    # Evaluate
    model.eval()
    phrase_ids, image_ids, pred_objs, true_objs, pred_scores, true_scores, img_kl_losses, img_xe_losses, masked_ixs = [], [], [], [], [], [], [], [], []
    for batch in tqdm(dl, total=len(dl)):
        image_id = batch[-1]
        batch = batch[:-1]
        if device.type != 'cpu':
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
        phrase_id, caption, input_mask, segment_ids, lm_label_ids, features, spatials, image_cls, \
        obj_labels, obj_confs, attr_labels, attr_confs, image_attrs, image_mask, image_labels = batch

        with torch.no_grad():
            _, prediction_scores_v_dict, _, _, _ = model(
                caption, features, spatials, 
                token_type_ids=segment_ids, attention_mask=input_mask, image_attention_mask=image_mask,
                masked_lm_labels=None, image_label=None, image_cls=image_cls, 
                obj_labels=obj_labels, obj_confs=obj_confs, attr_labels=attr_labels, 
                attr_confs=attr_confs, image_attrs=image_attrs
            )

            ix = list(config.visual_target_weights.keys())[0]
            if config.add_global_imgfeat == "last":
                prediction_scores_v = prediction_scores_v_dict[ix][:, :-1]
                image_label = image_labels[:, :-1]
            else:
                prediction_scores_v = prediction_scores_v_dict[ix][:, int(config.add_global_imgfeat is not None):]
                image_label = image_labels[:, int(config.add_global_imgfeat is not None):]

            target_ixs = [[] for _ in range(prediction_scores_v.size(0))]
            xs, ys = torch.where(image_label != -1)
            for x, y in zip(xs, ys):
                target_ixs[x].append(y.item())
            for bix in range(prediction_scores_v.size(0)):
                pred_bix_ixs, true_bix_ixs, bix_predictions, bix_labels = [], [], [], []
                for masked_ix in target_ixs[bix]:
                    pred_bix_ixs.append(torch.argmax(prediction_scores_v[bix, masked_ix]).item())
                    true_bix_ixs.append(torch.argmax(image_cls[bix, masked_ix]).item())
                    bix_predictions.append(prediction_scores_v[bix, masked_ix].numpy())
                    bix_labels.append(image_cls[bix, masked_ix].numpy())

                if ix == "0" or ix == "6":
                    masked_kl_loss = pre_vis_criterions[str(0)](prediction_scores_v[bix].unsqueeze(0), 1.0, image_label[bix].unsqueeze(0),
                                                                image_cls[bix].unsqueeze(0), features[bix].unsqueeze(0), obj_labels[bix].unsqueeze(0),
                                                                obj_confs[bix].unsqueeze(0), attr_labels[bix].unsqueeze(0), attr_confs[bix].unsqueeze(0))
                    masked_xe_loss = pre_vis_criterions[str(6)](prediction_scores_v[bix].unsqueeze(0), 1.0, image_label[bix].unsqueeze(0),
                                                                image_cls[bix].unsqueeze(0), features[bix].unsqueeze(0), obj_labels[bix].unsqueeze(0),
                                                                obj_confs[bix].unsqueeze(0), attr_labels[bix].unsqueeze(0), attr_confs[bix].unsqueeze(0))
                elif ix == "3":
                    masked_kl_loss = None
                    masked_xe_loss = pre_vis_criterions[str(3)](prediction_scores_v[bix].unsqueeze(0), 1.0, image_label[bix].unsqueeze(0),
                                                                image_cls[bix].unsqueeze(0), features[bix].unsqueeze(0), obj_labels[bix].unsqueeze(0),
                                                                obj_confs[bix].unsqueeze(0), attr_labels[bix].unsqueeze(0), attr_confs[bix].unsqueeze(0))
                if args.dump_results:
                    # pred_objs.append(pred_bix_ixs)
                    # true_objs.append(true_bix_ixs)
                    pred_scores.append(bix_predictions)
                    # true_scores.append(bix_labels)
                    # image_ids.append(image_id[bix].item())
                    # phrase_ids.append(phrase_id[bix].item())
                    img_kl_losses.append(masked_kl_loss)
                    img_xe_losses.append(masked_xe_loss)
                    # masked_ixs.append(target_ixs[bix])

    if default_gpu:
        print("Threshold: %.1f | Ablation: %s" % (args.overlap_threshold, args.masking))
        if masked_kl_loss is None:
            print("MRC-KL: None")
        else:
            print("MRC-KL:", np.mean(np.array(img_kl_losses)))
        print("MRC-XE:", np.mean(np.array(img_xe_losses)))

        if args.dump_results:
            eval_path = os.path.join(savePath, eval_split)
            # cPickle.dump(pred_objs, open(eval_path + "_%.1f_%s_preds.pkl" % (args.overlap_threshold, args.masking), "wb"))
            # cPickle.dump(true_objs, open(eval_path + "_%.1f_%s_gtobj.pkl" % (args.overlap_threshold, args.masking), "wb"))
            cPickle.dump(pred_scores, open(eval_path + "_%.1f_%s_score.pkl" % (args.overlap_threshold, args.masking), "wb"))
            # cPickle.dump(true_scores, open(eval_path + "_%.1f_%s_truth.pkl" % (args.overlap_threshold, args.masking), "wb"))
            # cPickle.dump(image_ids, open(eval_path + "_%.1f_%s_imgids.pkl" % (args.overlap_threshold, args.masking), "wb"))
            # cPickle.dump(phrase_ids, open(eval_path + "_%.1f_%s_phrids.pkl" % (args.overlap_threshold, args.masking), "wb"))
            cPickle.dump(img_kl_losses, open(eval_path + "_%.1f_%s_kld.pkl" % (args.overlap_threshold, args.masking), "wb"))
            cPickle.dump(img_xe_losses, open(eval_path + "_%.1f_%s_xe.pkl" % (args.overlap_threshold, args.masking), "wb"))
            # cPickle.dump(masked_ixs, open(eval_path + "_%.1f_%s_maskixs.pkl" % (args.overlap_threshold, args.masking), "wb"))


if __name__ == "__main__":
    main()
