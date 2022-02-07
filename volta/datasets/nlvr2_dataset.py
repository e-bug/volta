# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import jsonlines
import _pickle as cPickle

import base64
import numpy as np
import tensorpack.dataflow as td

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

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
        "question_id": item["question_id"],
        "image_id_0": item["image_id_0"],
        "image_id_1": item["image_id_1"],
        "sentence": item["sentence"],
        "answer": item,
    }
    return entry


def _load_dataset(dataroot, name):
    """Load entries

    dataroot: root path of dataset
    name: 'train', 'dev', 'test'
    """
    if name == "train" or name == "dev" or name == "test":
        annotations_path = os.path.join(dataroot, "%s.jsonl" % name)
        with jsonlines.open(annotations_path) as reader:
            # Build an index which maps image id with a list of hypothesis annotations.
            items = []
            count = 0
            for annotation in reader:
                dictionary = {}
                dictionary["id"] = annotation["identifier"]
                dictionary["image_id_0"] = (
                    "-".join(annotation["identifier"].split("-")[:-1]) + "-img0"
                )
                dictionary["image_id_1"] = (
                    "-".join(annotation["identifier"].split("-")[:-1]) + "-img1"
                )
                dictionary["question_id"] = count
                dictionary["sentence"] = str(annotation["sentence"])
                dictionary["labels"] = [0 if str(annotation["label"]) == "False" else 1]
                dictionary["scores"] = [1.0]
                items.append(dictionary)
                count += 1
    else:
        assert False, "data split is not recognized."

    entries = []
    for item in items:
        entries.append(_create_entry(item))
    return entries


def _load_marvl_dataset(annotations_path):
    """Load entries
    """
    with jsonlines.open(annotations_path) as reader:
        # Build an index which maps image id with a list of hypothesis annotations.
        items = []
        count = 0
        for annotation in reader:
            dictionary = {}
            dictionary["image_id_0"] = annotation["left_img"].split("/")[-1].split(".")[0]
            dictionary["image_id_1"] = annotation["right_img"].split("/")[-1].split(".")[0]
            dictionary["question_id"] = count
            dictionary["sentence"] = str(annotation["caption"])
            dictionary["labels"] = [int(annotation["label"])]
            dictionary["scores"] = [1.0]
            items.append(dictionary)
            count += 1

    entries = []
    for item in items:
        entries.append(_create_entry(item))
    return entries


class NLVR2Dataset(Dataset):
    def __init__(
        self,
        task,
        dataroot,
        annotations_jsonpath,
        split,
        image_features_reader,
        gt_image_features_reader,
        tokenizer,
        bert_model,
        padding_index=0,
        max_seq_length=16,
        max_region_num=37,
        num_locs=5,
        add_global_imgfeat=None,
        append_mask_sep=False,
    ):
        super().__init__()
        self.split = split
        self.num_labels = 2
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

        if annotations_jsonpath:
            self.entries = _load_marvl_dataset(annotations_jsonpath)
            self.tokenize(max_seq_length)
            self.tensorize()
        elif not os.path.exists(cache_path):
            self.entries = _load_dataset(dataroot, split)
            self.tokenize(max_seq_length)
            self.tensorize()
            if split in {"train", "dev"}:
                cPickle.dump(self.entries, open(cache_path, "wb"))
        elif split in {"train", "dev"}:
            logger.info("Loading from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb"))

    def tokenize(self, max_length=16):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in self.entries:
            tokens = self._tokenizer.encode(entry["sentence"])
            tokens = [tokens[0]] + tokens[1:-1][: max_length - 2] + [tokens[-1]]

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
        image_id_0 = entry["image_id_0"]
        image_id_1 = entry["image_id_1"]
        question_id = entry["question_id"]

        features_0, num_boxes_0, boxes_0, _ = self._image_features_reader[image_id_0]
        features_1, num_boxes_1, boxes_1, _ = self._image_features_reader[image_id_1]

        # mix_num_boxes = min(int(num_boxes_0) + int(num_boxes_1), self._max_region_num * 2)
        mix_boxes_pad = np.zeros((self._max_region_num * 2, self._num_locs))
        mix_features_pad = np.zeros((self._max_region_num * 2, 2048))

        image_mask = [1] * (num_boxes_0)
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)
        image_mask += [1] * (num_boxes_1)
        while len(image_mask) < 2 * self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:num_boxes_0] = boxes_0
        mix_boxes_pad[self._max_region_num: self._max_region_num+num_boxes_1] = boxes_1
        mix_features_pad[:num_boxes_0] = features_0
        mix_features_pad[self._max_region_num: self._max_region_num+num_boxes_1] = features_1

        img_segment_ids = np.zeros((mix_features_pad.shape[0]))
        img_segment_ids[:boxes_0.shape[0]] = 0
        img_segment_ids[boxes_0.shape[0]:] = 1

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        question = entry["q_token"]
        input_mask = entry["q_input_mask"]
        segment_ids = entry["q_segment_ids"]

        target = torch.zeros(self.num_labels)

        answer = entry["answer"]
        labels = answer["labels"]
        scores = answer["scores"]
        if labels is not None:
            target.scatter_(0, labels, scores)

        return features, spatials, image_mask, question, target, input_mask, segment_ids, question_id, index

    def __len__(self):
        return len(self.entries)


class NLVR2Loader(object):
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
        self.num_labels = 2
        self._max_region_num = max_region_num + int(add_global_imgfeat is not None)
        self._max_seq_length = max_seq_length
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self.num_locs = num_locs
        self.add_global_imgfeat = add_global_imgfeat
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
            num_locs=self.num_locs,
            padding_index=self._padding_index,
            add_global_imgfeat=self.add_global_imgfeat,
            norm_embeddings=self._norm_embeddings,
        )
        
        ds = td.PrefetchData(ds, cache, 1)
        ds = td.MapData(ds, preprocess_function)
        ds = td.PrefetchDataZMQ(ds, num_workers)
        self.ds = td.BatchData(ds, batch_size)
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers

    def __iter__(self):

        for ix, batch in enumerate(self.ds.get_data()):

            image_feats, image_locs, image_masks, \
                input_ids, input_mask, segment_ids, \
                labels, scores, \
                image_ids_0, image_ids_1, question_ids = batch

            batch_size = input_ids.shape[0]

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
        image_feat_0=None,
        image_feat_1=None,
        image_loc_0=None,
        image_loc_1=None,
        num_boxes_0=None,
        num_boxes_1=None,
        tokens=None,
        labels=None,
        scores=None,
    ):
        self.image_feat_0 = image_feat_0
        self.image_feat_1 = image_feat_1
        self.image_loc_0 = image_loc_0
        self.image_loc_1 = image_loc_1
        self.num_boxes_0 = num_boxes_0
        self.num_boxes_1 = num_boxes_1
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
            add_global_imgfeat=False,
            norm_embeddings=False,
    ):

        self.seq_len = seq_len
        self.region_len = region_len
        self.tokenizer = tokenizer
        self.num_caps = data_size
        self.bert_model = bert_model
        self.num_locs = num_locs
        self._padding_index = padding_index
        self.add_global_imgfeat = add_global_imgfeat
        self.norm_embeddings = norm_embeddings

    def __call__(self, item):

        try:
            features_0 = np.frombuffer(base64.b64decode(item["features_0"]), dtype=np.float32).reshape(-1, 2048)
            boxes_0 = np.frombuffer(base64.b64decode(item['boxes_0']), dtype=np.float32).reshape(-1, 4)
            features_1 = np.frombuffer(base64.b64decode(item["features_1"]), dtype=np.float32).reshape(-1, 2048)
            boxes_1 = np.frombuffer(base64.b64decode(item['boxes_1']), dtype=np.float32).reshape(-1, 4)
        except:
            features_0 = item["features_0"].reshape(-1, 2048)
            boxes_0 = item['boxes_0'].reshape(-1, 4)
            features_1 = item["features_1"].reshape(-1, 2048)
            boxes_1 = item['boxes_1'].reshape(-1, 4)

        image_location_0 = np.zeros((boxes_0.shape[0], self.num_locs), dtype=np.float32)
        image_location_0[:, :4] = boxes_0
        image_location_1 = np.zeros((boxes_1.shape[0], self.num_locs), dtype=np.float32)
        image_location_1[:, :4] = boxes_1

        image_w_0, image_h_0 = item['img_w_0'], item['img_h_0']
        image_w_1, image_h_1 = item['img_w_1'], item['img_h_1']
        if self.num_locs >= 5:
            image_location_0[:, -1] = (
                (image_location_0[:, 3] - image_location_0[:, 1])
                * (image_location_0[:, 2] - image_location_0[:, 0])
                / (float(image_w_0) * float(image_h_0))
            )
            image_location_1[:, -1] = (
                (image_location_1[:, 3] - image_location_1[:, 1])
                * (image_location_1[:, 2] - image_location_1[:, 0])
                / (float(image_w_1) * float(image_h_1))
            )
        image_location_0[:, 0] = image_location_0[:, 0] / float(image_w_0)
        image_location_0[:, 1] = image_location_0[:, 1] / float(image_h_0)
        image_location_0[:, 2] = image_location_0[:, 2] / float(image_w_0)
        image_location_0[:, 3] = image_location_0[:, 3] / float(image_h_0)
        
        image_location_1[:, 0] = image_location_1[:, 0] / float(image_w_1)
        image_location_1[:, 1] = image_location_1[:, 1] / float(image_h_1)
        image_location_1[:, 2] = image_location_1[:, 2] / float(image_w_1)
        image_location_1[:, 3] = image_location_1[:, 3] / float(image_h_1)

        if self.num_locs > 5:
            image_location_0[:, 4] = image_location_0[:, 2] - image_location_0[:, 0]
            image_location_0[:, 5] = image_location_0[:, 3] - image_location_0[:, 1]
            image_location_1[:, 4] = image_location_1[:, 2] - image_location_1[:, 0]
            image_location_1[:, 5] = image_location_1[:, 3] - image_location_1[:, 1]

        if self.norm_embeddings:
            features_0 = torch.FloatTensor(features_0.copy())
            features_0 = F.normalize(features_0, dim=-1).numpy()
            image_location_0 = image_location_0 / np.linalg.norm(image_location_0, 2, 1, keepdims=True)
            features_1 = torch.FloatTensor(features_1.copy())
            features_1 = F.normalize(features_1, dim=-1).numpy()
            image_location_1 = image_location_1 / np.linalg.norm(image_location_1, 2, 1, keepdims=True)

        num_boxes_0 = len(boxes_0)
        num_boxes_1 = len(boxes_1)

        if self.add_global_imgfeat == "first":
            g_feat = np.sum(features_0, axis=0) / num_boxes_0
            num_boxes_0 = num_boxes_0 + 1
            features_0 = np.concatenate([np.expand_dims(g_feat, axis=0), features_0], axis=0)
            g_location = [0, 0, 1, 1] + [1] * (self.num_locs - 4)
            image_location_0 = np.concatenate([np.expand_dims(g_location, axis=0), image_location_0], axis=0)

            g_feat = np.sum(features_1, axis=0) / num_boxes_1
            num_boxes_1 = num_boxes_1 + 1
            features_1 = np.concatenate([np.expand_dims(g_feat, axis=0), features_1], axis=0)
            g_location = [0, 0, 1, 1] + [1] * (self.num_locs - 4)
            image_location_1 = np.concatenate([np.expand_dims(g_location, axis=0), image_location_1], axis=0)

        elif self.add_global_imgfeat == "last":
            g_feat = np.sum(features_0, axis=0) / num_boxes_0
            num_boxes_0 = num_boxes_0 + 1
            features_0 = np.concatenate([features_0, np.expand_dims(g_feat, axis=0)], axis=0)
            g_location = [0, 0, 1, 1] + [1] * (self.num_locs - 4)
            image_location_0 = np.concatenate([image_location_0, np.expand_dims(g_location, axis=0)], axis=0)

            g_feat = np.sum(features_1, axis=0) / num_boxes_1
            num_boxes_1 = num_boxes_1 + 1
            features_1 = np.concatenate([features_1, np.expand_dims(g_feat, axis=0)], axis=0)
            g_location = [0, 0, 1, 1] + [1] * (self.num_locs - 4)
            image_location_1 = np.concatenate([image_location_1, np.expand_dims(g_location, axis=0)], axis=0)
        
        image_id_0 = item["img_id_0"]
        image_id_1 = item["img_id_1"]
        question_id = item["question_id"]
        tokens = self.tokenizer.encode(item["sentence"])
        tokens = [tokens[0]] + tokens[1:-1][: self.seq_len - 2] + [tokens[-1]]

        cur_example = InputExample(
            image_feat_0=features_0,
            image_feat_1=features_1,
            image_loc_0=image_location_0,
            image_loc_1=image_location_1,
            num_boxes_0=num_boxes_0,
            num_boxes_1=num_boxes_1,
            tokens=tokens,
            labels=item["labels"],
            scores=item["scores"],
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
            image_id_0,
            image_id_1,
            question_id,
        )

        return cur_tensors

    def convert_example_to_features(self, example, max_seq_length, tokenizer, max_region_length):
        max_region_length = max_region_length + int(self.add_global_imgfeat is not None)

        image_feat_0 = example.image_feat_0
        image_feat_1 = example.image_feat_1
        image_loc_0 = example.image_loc_0
        image_loc_1 = example.image_loc_1
        num_boxes_0 = int(example.num_boxes_0)
        num_boxes_1 = int(example.num_boxes_1)
        tokens = example.tokens
        labels = np.array(example.labels)
        scores = np.array(example.scores)
        
        input_ids = tokens
        segment_ids = [0] * len(tokens)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * len(input_ids)
        image_mask = [1] * num_boxes_0
        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_region_length:
            image_mask.append(0)
        image_mask += [1] * num_boxes_1
        while len(image_mask) < 2 * max_region_length:
            image_mask.append(0)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(self._padding_index)
            input_mask.append(0)
            segment_ids.append(0)

        image_loc = np.zeros((max_region_length * 2, self.num_locs))
        image_feat = np.zeros((max_region_length * 2, 2048))
        image_loc[:num_boxes_0] = image_loc_0
        image_loc[max_region_length: max_region_length+num_boxes_1] = image_loc_1
        image_feat[:num_boxes_0] = image_feat_0
        image_feat[max_region_length: max_region_length+num_boxes_1] = image_feat_1

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(image_mask) == 2 * max_region_length

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
