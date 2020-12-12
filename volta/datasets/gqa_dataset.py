# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import logging
import _pickle as cPickle

import numpy as np

import torch
from torch.utils.data import Dataset

from pytorch_transformers.tokenization_bert import BertTokenizer
from ._image_features_reader import ImageFeaturesH5Reader


logger = logging.getLogger(__name__)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _create_entry(item):
    entry = {
        "question_id": int(item["question_id"]),
        "image_id": item["image_id"],
        "question": item["question"],
        "answer": item,
    }
    return entry


def _load_dataset(dataroot, name):
    """Load entries

    dataroot: root path of dataset
    name: 'train', 'val', 'trainval', 'test'
    """
    if name == "train" or name == "val":
        items_path = os.path.join(dataroot, "%s_target.pkl" % name)
        items = cPickle.load(open(items_path, "rb"))
        items = sorted(items, key=lambda x: x["question_id"])
    elif name == "trainval":
        items_path = os.path.join(dataroot, "%s_target.pkl" % name)
        items = cPickle.load(open(items_path, "rb"))
        items = sorted(items, key=lambda x: x["question_id"])
        items = items[:-3000]
    elif name == "minval":
        items_path = os.path.join(dataroot, "trainval_target.pkl")
        items = cPickle.load(open(items_path, "rb"))
        items = sorted(items, key=lambda x: x["question_id"])
        items = items[-3000:]
    elif name == "test":
        items_path = os.path.join(dataroot, "testdev_balanced_questions.json")
        items = json.load(open(items_path, "rb"))
    else:
        assert False, "data split is not recognized."

    if "test" in name:
        entries = []
        for item in items:
            it = items[item]
            entry = {
                "question_id": int(item),
                "image_id": it["imageId"],
                "question": it["question"],
            }
            entries.append(entry)
    else:
        entries = []
        for item in items:
            entries.append(_create_entry(item))
    return entries


class GQAClassificationDataset(Dataset):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader,
        gt_image_features_reader: ImageFeaturesH5Reader,
        tokenizer: BertTokenizer,
        bert_model,
        padding_index: int = 0,
        max_seq_length: int = 16,
        max_region_num: int = 37,
        num_locs=5,
        add_global_imgfeat=None,
        append_mask_sep=False,
    ):
        super().__init__()
        self.split = split
        ans2label_path = os.path.join(dataroot, "trainval_ans2label.pkl")
        label2ans_path = os.path.join(dataroot, "trainval_label2ans.pkl")
        self.ans2label = cPickle.load(open(ans2label_path, "rb"))
        self.label2ans = cPickle.load(open(label2ans_path, "rb"))
        self.num_labels = len(self.ans2label)
        self._max_region_num = max_region_num + int(add_global_imgfeat is not None)
        self._max_seq_length = max_seq_length
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self._num_locs = num_locs
        self._add_global_imgfeat = add_global_imgfeat

        if "roberta" in bert_model:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task
                + "_"
                + split
                + "_"
                + "roberta"
                + "_"
                + str(max_seq_length)
                + ".pkl",
            )
        else:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task
                + "_"
                + split
                + "_"
                + str(max_seq_length)
                + ".pkl",
            )

        if not os.path.exists(cache_path):
            self.entries = _load_dataset(dataroot, split)
            self.tokenize(max_seq_length)
            self.tensorize()
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            logger.info("Loading from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb"))

        self.qid2imgid = {e["question_id"]: e["image_id"] for e in self.entries}

    def tokenize(self, max_length=16):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in self.entries:
            tokens = self._tokenizer.encode(entry["question"])
            tokens = tokens[: max_length - 2]
            tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), max_length)
            entry["q_token"] = tokens
            entry["q_input_mask"] = input_mask
            entry["q_segment_ids"] = segment_ids

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

            q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
            entry["q_input_mask"] = q_input_mask

            q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
            entry["q_segment_ids"] = q_segment_ids

            if "test" not in self.split:
                answer = entry["answer"]
                labels = np.array(answer["labels"])
                scores = np.array(answer["scores"], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry["answer"]["labels"] = labels
                    entry["answer"]["scores"] = scores
                else:
                    entry["answer"]["labels"] = None
                    entry["answer"]["scores"] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        image_id = entry["image_id"]
        question_id = entry["question_id"]
        features, num_boxes, boxes, _ = self._image_features_reader[image_id]

        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, self._num_locs))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        question = entry["q_token"]
        input_mask = entry["q_input_mask"]
        segment_ids = entry["q_segment_ids"]

        target = torch.zeros(self.num_labels)

        if "test" not in self.split:
            answer = entry["answer"]
            labels = answer["labels"]
            scores = answer["scores"]
            if labels is not None:
                target.scatter_(0, labels, scores)

        return features, spatials, image_mask, question, target, input_mask, segment_ids, question_id

    def __len__(self):
        return len(self.entries)
