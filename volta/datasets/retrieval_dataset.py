# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import random
import logging
import jsonlines
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


def _load_annotations(annotations_jsonpath, task):
    with jsonlines.open(annotations_jsonpath) as reader:
        # Build an index which maps image id with a list of caption annotations.
        entries = []
        imgid2entry = {}
        for annotation in reader:
            if task == "RetrievalCOCO":
                image_id = annotation["id"]
            elif task == "RetrievalFlickr30k":
                image_id = int(annotation["img_path"].split(".")[0])
            imgid2entry[image_id] = []
            count = 0
            for sentences in annotation["sentences"]:
                entries.append({"caption": sentences, "image_id": image_id})
                imgid2entry[image_id].append(count)
                count += 1
    return entries, imgid2entry


class RetrievalDataset(Dataset):
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
        max_seq_length: int = 20,
        max_region_num: int = 36,
        num_locs=5,
        add_global_imgfeat=None,
        append_mask_sep=False,
    ):
        # All the keys in `self._entries` would be present in `self._image_features_reader`
        self._entries, self.imgid2entry = _load_annotations(annotations_jsonpath, task)
        self.image_id_list = [*self.imgid2entry]

        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self.num_labels = 1
        self._split = split
        self._padding_index = padding_index
        self._max_region_num = max_region_num + int(add_global_imgfeat is not None)
        self._max_seq_length = max_seq_length
        self._num_locs = num_locs
        self._add_global_imgfeat = add_global_imgfeat

        if self._split == "train":
            image_info = cPickle.load(open(os.path.join(dataroot, "hard_negative" + ".pkl"), "rb"))
            for key, value in image_info.items():
                setattr(self, key, value)
            self.train_imgId2pool = {imageId: i for i, imageId in enumerate(self.train_image_list)}

        os.makedirs(os.path.join("/".join(annotations_jsonpath.split("/")[:-1]), "cache"), exist_ok=True)
        cache_path = os.path.join(
            "/".join(annotations_jsonpath.split("/")[:-1]), "cache",
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
            self.tokenize()
            self.tensorize()
            cPickle.dump(self._entries, open(cache_path, "wb"))
        else:
            print("loading entries from %s" % cache_path)
            self._entries = cPickle.load(open(cache_path, "rb"))

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self._entries:

            tokens = self._tokenizer.encode(entry["caption"])
            tokens = [tokens[0]] + tokens[1:-1][: self._max_seq_length - 2] + [tokens[-1]]

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < self._max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                tokens = tokens + padding
                input_mask += [0] * len(padding)
                segment_ids += [0] * len(padding)

            assert_eq(len(tokens), self._max_seq_length)
            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids

    def tensorize(self):
        for entry in self._entries:
            token = torch.from_numpy(np.array(entry["token"]))
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids

    def __getitem__(self, index):
        entry = self._entries[index]
        image_id = entry["image_id"]

        features, num_boxes, boxes, _ = self._image_features_reader[image_id]

        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, self._num_locs))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features1 = torch.tensor(mix_features_pad).float()
        image_mask1 = torch.tensor(image_mask).long()
        spatials1 = torch.tensor(mix_boxes_pad).float()

        caption1 = entry["token"]
        input_mask1 = entry["input_mask"]
        segment_ids1 = entry["segment_ids"]
        # negative samples.
        # 1: correct one, 2: random caption wrong, 3: random image wrong. 4: hard image wrong.

        while True:
            # sample a random image:
            img_id2 = random.choice(self.image_id_list)
            if img_id2 != image_id:
                entry2 = self._entries[random.choice(self.imgid2entry[img_id2])]
                break
            elif len(self.image_id_list) == 1:
                tokens = self._tokenizer.encode("[MASK]")
                segment_ids = [0] * len(tokens)
                input_mask = [1] * len(tokens)
                padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                tokens = torch.from_numpy(np.array(tokens + padding))
                input_mask = torch.from_numpy(np.array(input_mask + [0]*len(padding)))
                segment_ids = torch.from_numpy(np.array(segment_ids + [0]*len(padding)))
                entry2 = {"token": tokens, "input_mask": input_mask, "segment_ids": segment_ids}
                break
        
        features2 = features1
        image_mask2 = image_mask1
        spatials2 = spatials1
        caption2 = entry2["token"]
        input_mask2 = entry2["input_mask"]
        segment_ids2 = entry2["segment_ids"]

        # random image wrong
        while True:
            # sample a random image:
            img_id3 = random.choice(self.image_id_list)
            if img_id3 != image_id:
                break
            elif len(self.image_id_list) == 1:
                img_id3 = random.choice(self._image_features_reader._image_ids).decode()
                break

        features3, num_boxes3, boxes3, _ = self._image_features_reader[img_id3]
        image_mask3 = [1] * (int(num_boxes3))

        mix_num_boxes3 = min(int(num_boxes3), self._max_region_num)

        while len(image_mask3) < self._max_region_num:
            image_mask3.append(0)

        mix_boxes_pad[:mix_num_boxes3] = boxes3[:mix_num_boxes3]
        mix_features_pad[:mix_num_boxes3] = features3[:mix_num_boxes3]

        features3 = torch.tensor(mix_features_pad).float()
        image_mask3 = torch.tensor(image_mask3).long()
        spatials3 = torch.tensor(mix_boxes_pad).float()

        caption3 = caption1
        input_mask3 = input_mask1
        segment_ids3 = segment_ids1

        if self._split == "train":
            # random hard caption.
            rand_img_id_pool = self.train_hard_pool[self.train_imgId2pool[image_id]]
            pool_img_idx = int(rand_img_id_pool[np.random.randint(1, len(rand_img_id_pool))])
            img_id4 = self.train_image_list[pool_img_idx]
            entry4 = self._entries[random.choice(self.imgid2entry[img_id4])]
        else:
            while True:
                # sample a random image:
                img_id4 = random.choice(self.image_id_list)
                if img_id4 != image_id:
                    entry4 = self._entries[random.choice(self.imgid2entry[img_id4])]
                    break
                elif len(self.image_id_list) == 1:
                    tokens = self._tokenizer.encode("[MASK]")
                    segment_ids = [0] * len(tokens)
                    input_mask = [1] * len(tokens)
                    padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                    tokens = torch.from_numpy(np.array(tokens + padding))
                    input_mask = torch.from_numpy(np.array(input_mask + [0]*len(padding)))
                    segment_ids = torch.from_numpy(np.array(segment_ids + [0]*len(padding)))
                    entry4 = {"token": tokens, "input_mask": input_mask, "segment_ids": segment_ids} 
                    break

        features4 = features1
        image_mask4 = image_mask1
        spatials4 = spatials1
        caption4 = entry4["token"]
        input_mask4 = entry4["input_mask"]
        segment_ids4 = entry4["segment_ids"]

        features = torch.stack([features1, features2, features3, features4], dim=0)
        spatials = torch.stack([spatials1, spatials2, spatials3, spatials4], dim=0)
        image_mask = torch.stack([image_mask1, image_mask2, image_mask3, image_mask4], dim=0)
        caption = torch.stack([caption1, caption2, caption3, caption4], dim=0)
        input_mask = torch.stack([input_mask1, input_mask2, input_mask3, input_mask4], dim=0)
        segment_ids = torch.stack([segment_ids1, segment_ids2, segment_ids3, segment_ids4], dim=0)
        target = 0

        return features, spatials, image_mask, caption, target, input_mask, segment_ids, image_id, index

    def __len__(self):
        return len(self._entries)


def _load_annotationsVal(annotations_jsonpath, task):
    with jsonlines.open(annotations_jsonpath) as reader:
        # Build an index which maps image id with a list of caption annotations.
        image_entries = {}
        caption_entries = []
        for annotation in reader:
            if task == "RetrievalCOCO":
                image_id = annotation["id"]
            elif task == "RetrievalFlickr30k":
                image_id = int(annotation["img_path"].split(".")[0])
            elif task == "RetrievalxFlickrCO":
                image_id = annotation["img_path"]
            elif task == "RetrievalWIT":
                image_id = annotation["wit_ix"]
            image_entries[image_id] = 1
            if task == "RetrievalWIT":
                caption_entries.append({"caption": annotation["caption_reference_description"], "image_id": image_id})
            else:
                for sentences in annotation["sentences"]:
                    caption_entries.append({"caption": sentences, "image_id": image_id})
    image_entries = [*image_entries]
    return image_entries, caption_entries


class RetrievalDatasetVal(Dataset):
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
        max_seq_length: int = 20,
        max_region_num: int = 36,
        num_locs=5,
        add_global_imgfeat=None,
        append_mask_sep=False,
        num_subiters=2,
    ):
        # All the keys in `self._entries` would be present in `self._image_features_reader`
        self._image_entries, self._caption_entries = _load_annotationsVal(annotations_jsonpath, task)
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer

        self._split = split
        self._padding_index = padding_index
        self._max_region_num = max_region_num + int(add_global_imgfeat is not None)
        self._max_seq_length = max_seq_length
        self._num_locs = num_locs
        self._add_global_imgfeat = add_global_imgfeat
        self.num_labels = 1

        self.num_subiters = num_subiters
        self.num_images = len(self._image_entries)
        self.num_entries = len(self._caption_entries)
        self.max_num_images = self.num_images // self.num_subiters + int(self.num_images % self.num_subiters > 0)

        os.makedirs(os.path.join("/".join(annotations_jsonpath.split("/")[:-1]), "cache"), exist_ok=True)
        cache_path = os.path.join(
            "/".join(annotations_jsonpath.split("/")[:-1]),
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
            self.tokenize()
            self.tensorize()
            cPickle.dump(self._caption_entries, open(cache_path, "wb"))
        else:
            print("loading entries from %s" % cache_path)
            self._caption_entries = cPickle.load(open(cache_path, "rb"))

        self.features_all = np.zeros((len(self._image_entries), self._max_region_num, 2048))
        self.spatials_all = np.zeros((len(self._image_entries), self._max_region_num, self._num_locs))
        self.image_mask_all = np.zeros((len(self._image_entries), self._max_region_num))

        for i, image_id in enumerate(self._image_entries):
            features, num_boxes, boxes, _ = self._image_features_reader[image_id]

            mix_num_boxes = min(int(num_boxes), self._max_region_num)
            mix_boxes_pad = np.zeros((self._max_region_num, self._num_locs))
            mix_features_pad = np.zeros((self._max_region_num, 2048))

            image_mask = [1] * (int(mix_num_boxes))
            while len(image_mask) < self._max_region_num:
                image_mask.append(0)

            mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
            mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

            self.features_all[i] = mix_features_pad
            self.image_mask_all[i] = np.array(image_mask)
            self.spatials_all[i] = mix_boxes_pad

            sys.stdout.write("%d/%d\r" % (i, len(self._image_entries)))
            sys.stdout.flush()

        self.features_all = torch.Tensor(self.features_all).float()
        self.image_mask_all = torch.Tensor(self.image_mask_all).long()
        self.spatials_all = torch.Tensor(self.spatials_all).float()

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self._caption_entries:
            tokens = self._tokenizer.encode(entry["caption"])
            tokens = [tokens[0]] + tokens[1:-1][: self._max_seq_length - 2] + [tokens[-1]]

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < self._max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                tokens = tokens + padding
                input_mask += [0] * len(padding)
                segment_ids += [0] * len(padding)

            assert_eq(len(tokens), self._max_seq_length)
            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids

    def tensorize(self):
        for entry in self._caption_entries:
            token = torch.from_numpy(np.array(entry["token"])).long()
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"])).long()
            entry["segment_ids"] = segment_ids

    def __getitem__(self, index):
        # we iterate through every caption here.
        caption_idx = int(index / self.num_subiters)
        image_idx = index % self.num_subiters

        image_entries = self._image_entries[self.max_num_images * (image_idx):self.max_num_images * (image_idx + 1)]
        features_all = self.features_all[self.max_num_images * (image_idx):self.max_num_images * (image_idx + 1)]
        spatials_all = self.spatials_all[self.max_num_images * (image_idx):self.max_num_images * (image_idx + 1)]
        image_mask_all = self.image_mask_all[self.max_num_images * (image_idx):self.max_num_images * (image_idx + 1)]

        entry = self._caption_entries[caption_idx]
        caption = entry["token"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]

        target_all = torch.zeros(len(image_entries))
        for i, image_id in enumerate(image_entries):
            if image_id == entry["image_id"]:
                target_all[i] = 1

        return (
            features_all,
            spatials_all,
            image_mask_all,
            caption,
            input_mask,
            segment_ids,
            target_all,
            caption_idx,
            image_idx,
        )

    def __len__(self):
        return len(self._caption_entries) * self.num_subiters


class RetrievalLoader(object):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader, # features_path
        gt_image_features_reader,
        tokenizer,
        bert_model,
        padding_index: int = 0,
        max_seq_length: int = 16, # seq_len,
        max_region_num: int = 36, # reg_len,
        num_locs=5,
        add_global_imgfeat=None,
        append_mask_sep=False,
        norm_embeddings=False,
        batch_size=512,
        num_workers=25,
        cache=10000,
    ):
        self.split = split
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
        caption_path = annotations_jsonpath

        preprocess_function = BertPreprocessBatch(
            caption_path,
            tokenizer,
            bert_model,
            max_seq_length,
            max_region_num,
            self.num_dataset,
            num_locs=num_locs,
            padding_index=padding_index,
            norm_embeddings=self._norm_embeddings,
        )

        if split == "train":
            ds = td.PrefetchData(ds, cache, 1)
        ds = td.MapData(ds, preprocess_function)
        if split == "train":
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
                input_ids1s, input_mask1s, segment_ids1s, \
                input_ids3s, input_mask3s, segment_ids3s, \
                input_ids4s, input_mask4s, segment_ids4s, \
                image_ids = batch

            batch_size = input_ids1s.shape[0]

            if self.add_global_imgfeat == "first":
                sum_count = np.sum(image_masks == 1, axis=1, keepdims=True)
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
                g_image_feats = np.sum(image_feats, axis=1) / sum_count
                image_feats = np.concatenate([image_feats, np.expand_dims(g_image_feats, axis=1)], axis=1)
                image_feats = np.array(image_feats, dtype=np.float32)

                g_loc = [0, 0, 1, 1] + [1]*(self.num_locs - 4)
                g_image_locs = np.repeat(np.array([g_loc], dtype=np.float32), batch_size, axis=0)
                image_locs = np.concatenate([image_locs, np.expand_dims(g_image_locs, axis=1)], axis=1)

                image_locs = np.array(image_locs, dtype=np.float32)
                g_image_masks = np.repeat(np.array([[1]]), batch_size, axis=0)
                image_masks = np.concatenate([image_masks, g_image_masks], axis=1)

            # from pudb import set_trace; set_trace()

            image_feats = np.repeat(image_feats[:, np.newaxis], 4, axis=1)
            image_locs = np.repeat(image_locs[:, np.newaxis], 4, axis=1)
            image_masks = np.repeat(image_masks[:, np.newaxis], 4, axis=1)
            for i in range(0, batch_size):
                image_feats[i][1] = image_feats[(i+1)%batch_size][0]
                image_locs[i][1] = image_locs[(i+1)%batch_size][0]
                image_masks[i][1] = image_masks[(i+1)%batch_size][0]
            image_feats = torch.tensor(image_feats, dtype=torch.float)
            image_locs = torch.tensor(image_locs, dtype=torch.float)
            image_masks = torch.tensor(image_masks, dtype=torch.long)

            input_ids = np.stack([input_ids1s, input_ids1s, input_ids3s, input_ids4s], axis=1)
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            input_mask = np.stack([input_mask1s, input_mask1s, input_mask3s, input_mask4s], axis=1)
            input_mask = torch.tensor(input_mask, dtype=torch.long)
            segment_ids = np.stack([segment_ids1s, segment_ids1s, segment_ids3s, segment_ids4s], axis=1)
            segment_ids = torch.tensor(segment_ids, dtype=torch.long)

            target = torch.zeros(batch_size, dtype=torch.long)

            data = (
                image_feats,
                image_locs,
                image_masks,
                input_ids,
                target,
                input_mask, 
                segment_ids, 
                torch.tensor([int(i.split('_')[-1]) for i in image_ids]),
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
        caption1=None,
        caption3=None,
        caption4=None,
    ):
        self.image_feat = image_feat
        self.image_loc = image_loc
        self.num_boxes = num_boxes
        self.caption1 = caption1
        self.caption3 = caption3
        self.caption4 = caption4


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        image_feat=None,
        image_loc=None,
        image_mask=None,
        input_ids1=None,
        input_mask1=None,
        segment_ids1=None,
        input_ids3=None,
        input_mask3=None,
        segment_ids3=None,
        input_ids4=None,
        input_mask4=None,
        segment_ids4=None,
    ):
        self.image_feat = image_feat
        self.image_loc = image_loc
        self.image_mask = image_mask
        self.input_ids1 = input_ids1
        self.input_mask1 = input_mask1
        self.segment_ids1 = segment_ids1
        self.input_ids3 = input_ids3
        self.input_mask3 = input_mask3
        self.segment_ids3 = segment_ids3
        self.input_ids4 = input_ids4
        self.input_mask4 = input_mask4
        self.segment_ids4 = segment_ids4


class BertPreprocessBatch(object):
    def __init__(
            self,
            caption_path,
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

        self.captions = dict()
        self.img_ids = []
        with jsonlines.open(caption_path) as reader:
            for item in reader:
                self.captions[item['wit_ix']] = item['caption_reference_description']
                self.img_ids.append(item['wit_ix'])

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
        caption1 = self.captions[image_id]
        caption3 = self.random_cap(image_id)
        caption4 = self.random_cap(image_id)
        tokens_caption1 = self.tokenizer.encode(caption1)
        tokens_caption1 = [tokens_caption1[0]] + tokens_caption1[1:-1][: self.seq_len - 2] + [tokens_caption1[-1]]
        tokens_caption3 = self.tokenizer.encode(caption3)
        tokens_caption3 = [tokens_caption3[0]] + tokens_caption3[1:-1][: self.seq_len - 2] + [tokens_caption3[-1]]
        tokens_caption4 = self.tokenizer.encode(caption4)
        tokens_caption4 = [tokens_caption4[0]] + tokens_caption4[1:-1][: self.seq_len - 2] + [tokens_caption4[-1]]

        cur_example = InputExample(
            image_feat=image_feature,
            image_loc=image_location,
            num_boxes=num_boxes,
            caption1=tokens_caption1,
            caption3=tokens_caption3,
            caption4=tokens_caption4,
        )

        # transform sample to features
        cur_features = self.convert_example_to_features(cur_example, self.seq_len, self.tokenizer, self.region_len)

        cur_tensors = (
            cur_features.image_feat,
            cur_features.image_loc,
            cur_features.image_mask,
            cur_features.input_ids1,
            cur_features.input_mask1,
            cur_features.segment_ids1,
            cur_features.input_ids3,
            cur_features.input_mask3,
            cur_features.segment_ids3,
            cur_features.input_ids4,
            cur_features.input_mask4,
            cur_features.segment_ids4,
            image_id,
        )
        return cur_tensors

    def random_cap(self, img_id):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        rnd_img_id = self.img_ids[random.randint(0, self.num_caps - 1)]
        while rnd_img_id == img_id:
            rnd_img_id = self.img_ids[random.randint(0, self.num_caps - 1)]
        return self.captions[rnd_img_id]

    def convert_example_to_features(self, example, max_seq_length, tokenizer, max_region_length):

        image_feat = example.image_feat
        image_loc = example.image_loc
        num_boxes = int(example.num_boxes)
        tokens1 = example.caption1
        tokens3 = example.caption3
        tokens4 = example.caption4

        segment_ids1 = [0] * len(tokens1)
        segment_ids3 = [0] * len(tokens3)
        segment_ids4 = [0] * len(tokens4)

        input_ids1 = tokens1
        input_ids3 = tokens3
        input_ids4 = tokens4

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask1 = [1] * len(input_ids1)
        input_mask3 = [1] * len(input_ids3)
        input_mask4 = [1] * len(input_ids4)
        image_mask = [1] * num_boxes
        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_region_length:
            image_mask.append(0)

        # Zero-pad up to the sequence length.
        while len(input_ids1) < max_seq_length:
            input_ids1.append(self._padding_index)
            input_mask1.append(0)
            segment_ids1.append(0)
        while len(input_ids3) < max_seq_length:
            input_ids3.append(self._padding_index)
            input_mask3.append(0)
            segment_ids3.append(0)
        while len(input_ids4) < max_seq_length:
            input_ids4.append(self._padding_index)
            input_mask4.append(0)
            segment_ids4.append(0)

        assert len(input_ids1) == max_seq_length
        assert len(input_mask1) == max_seq_length
        assert len(segment_ids1) == max_seq_length
        assert len(image_mask) == max_region_length

        features = InputFeatures(
            image_feat=image_feat,
            image_loc=image_loc,
            image_mask=np.array(image_mask),
            input_ids1=np.array(input_ids1),
            input_mask1=np.array(input_mask1),
            segment_ids1=np.array(segment_ids1),
            input_ids3=np.array(input_ids3),
            input_mask3=np.array(input_mask3),
            segment_ids3=np.array(segment_ids3),
            input_ids4=np.array(input_ids4),
            input_mask4=np.array(input_mask4),
            segment_ids4=np.array(segment_ids4),
        )
        return features
