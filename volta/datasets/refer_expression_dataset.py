# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import _pickle as cPickle

import numpy as np

import torch
from torch.utils.data import Dataset

from pytorch_transformers.tokenization_bert import BertTokenizer
from ._image_features_reader import ImageFeaturesH5Reader

from tools.refer.refer import REFER


def iou(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = (
        (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    ).view(1, K)

    anchors_area = (
        (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
    ).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (
        torch.min(boxes[:, :, 2], query_boxes[:, :, 2])
        - torch.max(boxes[:, :, 0], query_boxes[:, :, 0])
        + 1
    )
    iw[iw < 0] = 0

    ih = (
        torch.min(boxes[:, :, 3], query_boxes[:, :, 3])
        - torch.max(boxes[:, :, 1], query_boxes[:, :, 1])
        + 1
    )
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


class ReferExpressionDataset(Dataset):
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
        max_seq_length: int = 20,
        max_region_num: int = 60,
        num_locs=5,
        add_global_imgfeat=None,
        append_mask_sep=False,
    ):
        self.split = split

        if task == "refcocog":
            self.refer = REFER(dataroot, dataset=task, splitBy="umd")
        else:
            self.refer = REFER(dataroot, dataset=task, splitBy="unc")

        if self.split == "mteval":
            self.ref_ids = self.refer.getRefIds(split="train")
        else:
            self.ref_ids = self.refer.getRefIds(split=split)

        print("%s refs are in split [%s]." % (len(self.ref_ids), split))

        self.num_labels = 1
        self._image_features_reader = image_features_reader
        self._gt_image_features_reader = gt_image_features_reader
        self._tokenizer = tokenizer

        self._padding_index = padding_index
        self._max_seq_length = max_seq_length
        self.dataroot = dataroot
        self.entries = self._load_annotations()

        self._max_region_num = max_region_num + int(add_global_imgfeat is not None)
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
                + "_"
                + str(max_region_num)
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
                + "_"
                + str(max_region_num)
                + ".pkl",
            )

        if not os.path.exists(cache_path):
            self.tokenize()
            self.tensorize()
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            print("loading entries from %s" % (cache_path))
            self.entries = cPickle.load(open(cache_path, "rb"))

    def _load_annotations(self):
        # Build an index which maps image id with a list of caption annotations.
        entries = []
        remove_ids = []
        if self.split == "mteval":
            remove_ids = np.load(
                os.path.join(self.dataroot, "cache", "coco_test_ids.npy")
            )
            remove_ids = [int(x) for x in remove_ids]

        for ref_id in self.ref_ids:
            ref = self.refer.Refs[ref_id]
            image_id = ref["image_id"]
            if self.split == "train" and int(image_id) in remove_ids:
                continue
            elif self.split == "mteval" and int(image_id) not in remove_ids:
                continue
            ref_id = ref["ref_id"]
            refBox = self.refer.getRefBox(ref_id)
            for sent, sent_id in zip(ref["sentences"], ref["sent_ids"]):
                caption = sent["raw"]
                entries.append(
                    {
                        "caption": caption,
                        "sent_id": sent_id,
                        "image_id": image_id,
                        "refBox": refBox,
                        "ref_id": ref_id,
                    }
                )

        return entries

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self.entries:
            tokens = self._tokenizer.encode(entry["caption"])
            tokens = tokens[: self._max_seq_length - 2]
            tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < self._max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), self._max_seq_length)
            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids

    def tensorize(self):
        for entry in self.entries:
            token = torch.from_numpy(np.array(entry["token"]))
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids

    def __getitem__(self, index):
        entry = self.entries[index]

        image_id = entry["image_id"]
        ref_box = entry["refBox"]

        ref_box = [
            ref_box[0],
            ref_box[1],
            ref_box[0] + ref_box[2],
            ref_box[1] + ref_box[3],
        ]
        features, num_boxes, boxes, boxes_ori = self._image_features_reader[image_id]

        boxes_ori = boxes_ori[:num_boxes]
        boxes = boxes[:num_boxes]
        features = features[:num_boxes]

        mix_boxes_ori = boxes_ori
        mix_boxes = boxes
        mix_features = features
        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_target = iou(
            torch.tensor(mix_boxes_ori[:, :4]).float(),
            torch.tensor([ref_box]).float(),
        )

        image_mask = [1] * (mix_num_boxes)
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad = np.zeros((self._max_region_num, self._num_locs))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        mix_boxes_pad[:mix_num_boxes] = mix_boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = mix_features[:mix_num_boxes]

        # appending the target feature.
        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        target = torch.zeros((self._max_region_num, 1)).float()
        target[:mix_num_boxes] = mix_target[:mix_num_boxes]

        spatials_ori = torch.tensor(mix_boxes_ori).float()

        caption = entry["token"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]

        return features, spatials, image_mask, caption, target, input_mask, segment_ids, image_id

    def __len__(self):
        return len(self.entries)
