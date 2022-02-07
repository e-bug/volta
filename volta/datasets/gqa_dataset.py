# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import logging
import _pickle as cPickle

import base64
import numpy as np
import tensorpack.dataflow as td

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from transformers import AutoTokenizer
from ._image_features_reader import ImageFeaturesH5Reader

import msgpack_numpy
msgpack_numpy.patch()

MAX_MSGPACK_LEN = 1000000000

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


def _load_dataset(dataroot, name, annotations_jsonpath):
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
    elif name.startswith("test_"):
        items_path = os.path.join(annotations_jsonpath)
        items = json.load(open(items_path, "rb"))
    elif name.startswith("train_") or name.startswith("dev_"):
        items_path = annotations_jsonpath
        items = cPickle.load(open(items_path, "rb"))
        items = sorted(items, key=lambda x: x["question_id"])
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
        tokenizer: AutoTokenizer,
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

        cache_path = os.path.join(
            dataroot,
            "cache",
            task
            + "_"
            + split
            + "_"
            + bert_model.split("/")[-1]
            + "_"
            + str(max_seq_length)
            + ".pkl",
        )
        
        if not os.path.exists(cache_path):
            self.entries = _load_dataset(dataroot, split, annotations_jsonpath)
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
            tokens = [tokens[0]] + tokens[1:-1][: self._max_seq_length - 2] + [tokens[-1]]

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += [0] * len(padding)
                segment_ids += [0] * len(padding)

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

        return features, spatials, image_mask, question, target, input_mask, segment_ids, question_id, index

    def __len__(self):
        return len(self.entries)


class GQAClassificationLoader(object):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader,
        gt_image_features_reader,
        tokenizer,
        bert_model,
        padding_index: int = 0,
        max_seq_length: int = 16,
        max_region_num: int = 36,
        num_locs=5,
        add_global_imgfeat=None,
        append_mask_sep=False,
        norm_embeddings=False,
        batch_size=512,
        num_workers=25,
        cache=10000,
    ):
        self.split = split
        ans2label_path = os.path.join(dataroot, "trainval_ans2label.pkl")
        label2ans_path = os.path.join(dataroot, "trainval_label2ans.pkl")
        self.ans2label = cPickle.load(open(ans2label_path, "rb"))
        self.label2ans = cPickle.load(open(label2ans_path, "rb"))
        self.num_labels = len(self.ans2label)
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self._norm_embeddings = norm_embeddings

        lmdb_file = image_features_reader
        print("Loading from %s" % lmdb_file)

        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = len(ds)
        if split == "train":
            ds = td.LocallyShuffleData(ds, cache)

        preprocess_function = BertPreprocessBatch(
            tokenizer,
            bert_model,
            max_seq_length,
            max_region_num,
            self.num_dataset,
            num_locs=num_locs,
            padding_index=padding_index,
            norm_embeddings=self._norm_embeddings,
        )
        
        # if split == "train":
        ds = td.PrefetchData(ds, cache, 1)
        ds = td.MapData(ds, preprocess_function)
        # if split == "train":
        ds = td.PrefetchDataZMQ(ds, num_workers)
        self.ds = td.BatchData(ds, batch_size)
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.add_global_imgfeat = add_global_imgfeat
        self.num_locs = num_locs

    def __iter__(self):

        for ix, batch in enumerate(self.ds.get_data()):

            image_feats, image_locs, image_masks, \
                input_ids, input_mask, segment_ids, \
                labels, scores, \
                image_ids, question_ids = batch

            batch_size = input_ids.shape[0]

            if self.add_global_imgfeat == "first":
                sum_count = np.sum(image_masks == 1, axis=1, keepdims=True)
                # sum_count[sum_count == 0] = 1
                g_image_feats = np.sum(image_feats, axis=1) / sum_count
                image_feats = np.concatenate([np.expand_dims(g_image_feats, axis=1), image_feats], axis=1)
                image_feats = np.array(image_feats, dtype=np.float32)

                g_loc = [0, 0, 1, 1] + [1]*(self.num_locs - 4)
                g_image_locs = np.repeat(np.array([g_loc], dtype=np.float32), batch_size, axis=0)
                image_locs = np.concatenate([np.expand_dims(g_image_locs, axis=1), image_locs], axis=1)

                image_locs = np.array(image_locs, dtype=np.float32)
                g_image_masks = np.repeat(np.array([[1]]), batch_size, axis=0)
                image_masks = np.concatenate([g_image_masks, image_masks], axis=1)

            elif self.add_global_imgfeat == "last":
                sum_count = np.sum(image_masks == 1, axis=1, keepdims=True)
                # sum_count[sum_count == 0] = 1
                g_image_feats = np.sum(image_feats, axis=1) / sum_count
                image_feats = np.concatenate([image_feats, np.expand_dims(g_image_feats, axis=1)], axis=1)
                image_feats = np.array(image_feats, dtype=np.float32)

                g_loc = [0, 0, 1, 1] + [1]*(self.num_locs - 4)
                g_image_locs = np.repeat(np.array([g_loc], dtype=np.float32), batch_size, axis=0)
                image_locs = np.concatenate([image_locs, np.expand_dims(g_image_locs, axis=1)], axis=1)

                image_locs = np.array(image_locs, dtype=np.float32)
                g_image_masks = np.repeat(np.array([[1]]), batch_size, axis=0)
                image_masks = np.concatenate([image_masks, g_image_masks], axis=1)

            image_feats = torch.tensor(image_feats, dtype=torch.float)
            image_locs = torch.tensor(image_locs, dtype=torch.float)
            image_masks = torch.tensor(image_masks, dtype=torch.long)
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            input_mask = torch.tensor(input_mask, dtype=torch.long)
            segment_ids = torch.tensor(segment_ids, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
            scores = torch.tensor(scores, dtype=torch.float)
            target = torch.zeros((batch_size, self.num_labels), dtype=torch.float)
            if labels is not None:
                target.scatter_(1, labels, scores)

            data = (
                image_feats,
                image_locs,
                image_masks,
                input_ids,
                target,
                input_mask, 
                segment_ids, 
                torch.tensor(question_ids),
                torch.tensor(ix)
            )
            yield data

    def __len__(self):
        return self.ds.size()


class InputExample(object):
    def __init__(
        self,
        image_feat=None,
        image_loc=None,
        num_boxes=None,
        tokens=None,
        labels=None,
        scores=None,
    ):
        self.image_feat = image_feat
        self.image_loc = image_loc
        self.num_boxes = num_boxes
        self.tokens = tokens
        self.labels = labels
        self.scores = scores


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        image_feat=None,
        image_loc=None,
        image_mask=None,
        input_ids=None,
        input_mask=None,
        segment_ids=None,
        labels=None,
        scores=None,
    ):
        self.image_feat = image_feat
        self.image_loc = image_loc
        self.image_mask = image_mask
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.labels = labels
        self.scores = scores


class BertPreprocessBatch(object):
    def __init__(
            self,
            tokenizer,
            bert_model,
            seq_len,
            region_len,
            data_size,
            num_locs=5,
            padding_index=0,
            norm_embeddings=False,
    ):

        self.seq_len = seq_len
        self.region_len = region_len
        self.tokenizer = tokenizer
        self.num_caps = data_size
        self.bert_model = bert_model
        self.num_locs = num_locs
        self._padding_index = padding_index
        self.norm_embeddings = norm_embeddings

    def __call__(self, item):
        image_feature = np.zeros((self.region_len, 2048), dtype=np.float32)
        image_location = np.zeros((self.region_len, self.num_locs), dtype=np.float32)

        try:
            features = np.frombuffer(base64.b64decode(item["features"]), dtype=np.float32).reshape(-1, 2048)
            boxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(-1, 4)
        except:
            features = item["features"].reshape(-1, 2048)
            boxes = item['boxes'].reshape(-1, 4)

        num_boxes = len(boxes)
        image_location[:num_boxes, :4] = boxes

        image_w, image_h = item['img_w'], item['img_h']
        if self.num_locs >= 5:
            image_location[:, -1] = (
                (image_location[:, 3] - image_location[:, 1])
                * (image_location[:, 2] - image_location[:, 0])
                / (float(image_w) * float(image_h))
            )
 
        # Normalize the box locations (to 0 ~ 1)
        image_location[:, 0] = image_location[:, 0] / float(image_w)
        image_location[:, 1] = image_location[:, 1] / float(image_h)
        image_location[:, 2] = image_location[:, 2] / float(image_w)
        image_location[:, 3] = image_location[:, 3] / float(image_h)
        
        if self.num_locs > 5:
            image_location[:, 4] = image_location[:, 2] - image_location[:, 0]
            image_location[:, 5] = image_location[:, 3] - image_location[:, 1]

        if self.norm_embeddings:
            features = torch.FloatTensor(features.copy())
            features = F.normalize(features, dim=-1).numpy()
            image_location = image_location / np.linalg.norm(image_location, 2, 1, keepdims=True)

        image_feature[:num_boxes] = features

        image_id = item['img_id']
        entry = _create_entry(item["entry"])
        question_id = entry["question_id"]
        tokens = self.tokenizer.encode(entry["question"])
        tokens = [tokens[0]] + tokens[1:-1][: self.seq_len - 2] + [tokens[-1]]
        answer = entry["answer"]

        cur_example = InputExample(
            image_feat=image_feature,
            image_loc=image_location,
            num_boxes=num_boxes,
            tokens=tokens,
            labels=answer["labels"],
            scores=answer["scores"],
        )

        # transform sample to features
        cur_features = self.convert_example_to_features(cur_example, self.seq_len, self.tokenizer, self.region_len)

        cur_tensors = (
            cur_features.image_feat,
            cur_features.image_loc,
            cur_features.image_mask,
            cur_features.input_ids,
            cur_features.input_mask,
            cur_features.segment_ids,
            cur_features.labels,
            cur_features.scores,
            image_id,
            question_id,
        )
        return cur_tensors

    def convert_example_to_features(self, example, max_seq_length, tokenizer, max_region_length):

        image_feat = example.image_feat
        image_loc = example.image_loc
        num_boxes = int(example.num_boxes)
        tokens = example.tokens
        labels = np.array(example.labels)
        scores = np.array(example.scores)
        
        input_ids = tokens
        segment_ids = [0] * len(tokens)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * len(input_ids)
        image_mask = [1] * num_boxes
        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_region_length:
            image_mask.append(0)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(self._padding_index)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(image_mask) == max_region_length

        features = InputFeatures(
            image_feat=image_feat,
            image_loc=image_loc,
            image_mask=np.array(image_mask),
            input_ids=np.array(input_ids),
            input_mask=np.array(input_mask),
            segment_ids=np.array(segment_ids),
            labels=labels,
            scores=scores,
        )
        return features
