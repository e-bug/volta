# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from torch.utils.data import Dataset
import numpy as np

from pytorch_transformers.tokenization_bert import BertTokenizer
from ._all_image_features_reader import ImageFeaturesH5Reader
import _pickle as cPickle

import xml.etree.ElementTree as ET


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


def iot(anchors, gt_boxes):
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

    overlaps = iw * ih / gt_boxes_area

    return overlaps


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def get_sentence_data(fn):
    """
    Parses a sentence file from the Flickr30K Entities dataset
    input:
      fn - full file path to the sentence file to parse

    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this
                                    phrase belongs to
    """
    with open(fn, "r") as f:
        sentences = f.read().split("\n")

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue
        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == "]":
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(" ".join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == "[":
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split("/")
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {"sentence": " ".join(words), "phrases": []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data["phrases"].append(
                {
                    "first_word_index": index,
                    "phrase": phrase,
                    "phrase_id": p_id,
                    "phrase_type": p_type,
                }
            )
        annotations.append(sentence_data)

    return annotations


def get_annotations(fn):
    """
    Parses the xml files in the Flickr30K Entities dataset
    input:
      fn - full file path to the annotations file to parse
    output:
      dictionary with the following fields:
          scene - list of identifiers which were annotated as
                  pertaining to the whole scene
          nobox - list of identifiers which were annotated as
                  not being visible in the image
          boxes - a dictionary where the fields are identifiers
                  and the values are its list of boxes in the
                  [xmin ymin xmax ymax] format
    """
    tree = ET.parse(fn)
    root = tree.getroot()
    size_container = root.findall("size")[0]
    anno_info = {"boxes": {}, "scene": [], "nobox": []}
    for size_element in size_container:
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall("object"):
        for names in object_container.findall("name"):
            box_id = names.text
            box_container = object_container.findall("bndbox")
            if len(box_container) > 0:
                if box_id not in anno_info["boxes"]:
                    anno_info["boxes"][box_id] = []
                xmin = int(box_container[0].findall("xmin")[0].text) - 1
                ymin = int(box_container[0].findall("ymin")[0].text) - 1
                xmax = int(box_container[0].findall("xmax")[0].text) - 1
                ymax = int(box_container[0].findall("ymax")[0].text) - 1
                anno_info["boxes"][box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall("nobndbox")[0].text)
                if nobndbox > 0:
                    anno_info["nobox"].append(box_id)

                scene = int(object_container.findall("scene")[0].text)
                if scene > 0:
                    anno_info["scene"].append(box_id)

    return anno_info


def load_annotations(dataroot, split):
    with open(os.path.join(dataroot, "%s.txt" % split), "r") as f:
        images = f.read().splitlines()

    entries = []
    for img in images:
        annotation = get_annotations(os.path.join(dataroot, "Annotations", img + ".xml"))
        sentences = get_sentence_data(os.path.join(dataroot, "Sentences", img + ".txt"))
        for i, sent in enumerate(sentences):
            for phrase in sent["phrases"]:
                if str(phrase["phrase_id"]) in annotation["boxes"].keys():
                    entries.append(
                        {
                            "caption": phrase["phrase"],
                            "first_word_index": phrase["first_word_index"],
                            "sent_id": int(phrase["phrase_id"]),
                            "image_id": int(img),
                            "refBoxes": annotation["boxes"][str(phrase["phrase_id"])],
                            "sentence": sent["sentence"]
                        }
                    )
    return entries


class FlickrVis4LangDataset(Dataset):
    def __init__(
        self,
        task: str,
        dataroot: str,
        masking: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader,
        gt_image_features_reader: ImageFeaturesH5Reader,
        tokenizer: BertTokenizer,
        bert_model,
        padding_index: int = 0,
        max_seq_length: int = 20,
        max_region_num: int = 36,
        num_locs=5,
        threshold=0.5,
        add_global_imgfeat=None,
        append_mask_sep=False,
    ):
        self._entries = load_annotations(dataroot, split)

        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self.num_labels = 1
        self._split = split
        self._padding_index = padding_index
        self._max_region_num = max_region_num + int(add_global_imgfeat is not None)
        self._max_seq_length = max_seq_length
        self._num_locs = num_locs
        self._add_global_imgfeat = add_global_imgfeat
        self.dataroot = dataroot
        self.masking = masking
        self.threshold = threshold

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
            self.tokenize()
            self.tensorize()
            cPickle.dump(self._entries, open(cache_path, "wb"))
        else:
            print("loading entries from %s" % cache_path)
            self._entries = cPickle.load(open(cache_path, "rb"))

        self._avg_feature = torch.zeros(2048)
        for entry in self._entries:
            image_id = entry["image_id"]
            features, num_boxes, boxes, boxes_ori, image_cls, obj_labels, obj_confs, \
                attr_labels, attr_confs, image_attrs = self._image_features_reader[image_id]
            self._avg_feature += features.sum(0)
        self._avg_feature = self._avg_feature / features.shape[0] / len(self._entries)

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self._entries:
            # original
            phrase = entry['caption']
            phrase_id = entry['sent_id']
            start_ix = entry['first_word_index']
            sent = entry['sentence']

            # tokenized
            words = sent.lower().split()
            subphrase = self._tokenizer.tokenize(phrase)
            subwords = self._tokenizer.tokenize(sent)
            word_ix = 0
            word_rec = ''
            old2new_first_ixs = {}
            for subword_ix, subword in enumerate(subwords):
                sub = subword.lstrip('##')
                if not word_rec:
                    # new word
                    old2new_first_ixs[word_ix] = subword_ix
                word_rec += sub
                if word_rec == words[word_ix]:
                    # recovered
                    word_rec = ''
                    word_ix += 1

            # masked
            mask_sent = []
            lm_label_ids = []
            tomask = False
            phrase_rec = []
            for ix, tok in enumerate(subwords):
                if ix == old2new_first_ixs[start_ix] or tomask:
                    tomask = True
                    mask_sent.append(self._tokenizer.mask_token)
                    lm_label_ids.append(self._tokenizer.convert_tokens_to_ids(tok))
                    phrase_rec += [tok]
                    if phrase_rec == subphrase:
                        tomask = False
                else:
                    mask_sent.append(tok)
                    lm_label_ids.append(-1)

            mask_sent_ids = self._tokenizer.convert_tokens_to_ids(mask_sent)
            mask_sent_ids = mask_sent_ids[: self._max_seq_length - 2]
            mask_sent_ids = self._tokenizer.add_special_tokens_single_sentence(mask_sent_ids)

            lm_label_ids = lm_label_ids[: self._max_seq_length - 2]
            lm_label_ids = [-1] + lm_label_ids + [-1]
            segment_ids = [0] * len(mask_sent_ids)
            input_mask = [1] * len(mask_sent_ids)

            if len(mask_sent_ids) < self._max_seq_length:
                padding = [self._padding_index] * (self._max_seq_length - len(mask_sent_ids))
                mask_sent_ids = mask_sent_ids + padding
                input_mask += padding
                segment_ids += padding
                lm_label_ids += [-1] * len(padding)

            assert_eq(len(mask_sent_ids), self._max_seq_length)
            entry["token"] = mask_sent_ids
            entry["phrase_id"] = phrase_id
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids
            entry["lm_label_ids"] = lm_label_ids

    def tensorize(self):
        for entry in self._entries:
            token = torch.from_numpy(np.array(entry["token"]))
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids

            lm_label_ids = torch.from_numpy(np.array(entry["lm_label_ids"]))
            entry["lm_label_ids"] = lm_label_ids

            phrase_id = torch.from_numpy(np.array(entry["phrase_id"]))
            entry["phrase_id"] = phrase_id

    def __getitem__(self, index):
        entry = self._entries[index]
        image_id = entry["image_id"]
        ref_boxes = entry["refBoxes"]

        features, num_boxes, boxes, boxes_ori, image_cls, obj_labels, obj_confs, \
            attr_labels, attr_confs, image_attrs = self._image_features_reader[image_id]

        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, self._num_locs))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        # mask the overlap regions into zeros
        output_label = np.zeros((mix_features_pad.shape[0])) - 1
        masked_label = np.zeros((mix_features_pad.shape[0]))
        overlaps = iot(torch.tensor(boxes_ori[:, :4]).float(), torch.tensor(ref_boxes).float())
        overlap_threshold = 2.0  # no box can overlap more than 1.0 --> this leads to no ablated box
        if self.masking == "all":
            overlap_threshold = -1.0  # every box overlaps at least for 0.0 --> this leads to all box being ablated
        elif self.masking == "object":
            overlap_threshold = self.threshold
        masked_label = np.logical_or(masked_label, (overlaps > overlap_threshold).max(1)[0])
        if self._add_global_imgfeat == "first":
            masked_label[0] = 0
        elif self._add_global_imgfeat == "last":
            masked_label[-1] = 0
        # change token to ablation token
        mix_features_pad[masked_label > 0] = self._avg_feature
        
        output_label[masked_label > 0] = 1
        if self._add_global_imgfeat:  
            # set the [IMG] region to the average of all the other regions
            mix_features_pad[0] = mix_features_pad[1:].mean(0)

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        caption = entry["token"]
        phrase_id = entry["phrase_id"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]
        lm_label_ids = entry["lm_label_ids"]

        region_labels = torch.zeros_like(image_mask) - 1

        return (
            phrase_id,
            caption,
            input_mask,
            segment_ids,
            lm_label_ids,
            features,
            spatials,
            image_cls,
            obj_labels,
            obj_confs,
            attr_labels,
            attr_confs,
            image_attrs,
            image_mask,
            region_labels,
            image_id,
        )

    def __len__(self):
        return len(self._entries)


class FlickrLang4VisDataset(Dataset):
    def __init__(
        self,
        task: str,
        dataroot: str,
        masking: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader,
        gt_image_features_reader: ImageFeaturesH5Reader,
        tokenizer: BertTokenizer,
        bert_model,
        padding_index: int = 0,
        max_seq_length: int = 20,
        max_region_num: int = 36,
        num_locs=5,
        threshold=0.5,
        add_global_imgfeat=None,
        append_mask_sep=False,
    ):
        self._entries = load_annotations(dataroot, split)

        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self.num_labels = 1
        self._split = split
        self._padding_index = padding_index
        self._max_region_num = max_region_num + int(add_global_imgfeat is not None)
        self._max_seq_length = max_seq_length
        self._num_locs = num_locs
        self._add_global_imgfeat = add_global_imgfeat
        self.dataroot = dataroot
        self.masking = masking
        self.threshold = threshold

        self.tokenize()
        self.tensorize()

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self._entries:
            # original
            phrase = entry['caption']
            phrase_id = entry['sent_id']
            start_ix = entry['first_word_index']
            sent = entry['sentence']

            # tokenized
            words = sent.lower().split()
            subphrase = self._tokenizer.tokenize(phrase)
            subwords = self._tokenizer.tokenize(sent)
            word_ix = 0
            word_rec = ''
            old2new_first_ixs = {}
            for subword_ix, subword in enumerate(subwords):
                sub = subword.lstrip('##')
                if not word_rec:
                    # new word
                    old2new_first_ixs[word_ix] = subword_ix
                word_rec += sub
                if word_rec == words[word_ix]:
                    # recovered
                    word_rec = ''
                    word_ix += 1

            # masked
            mask_sent = []
            lm_label_ids = []
            if self.masking == "all":
                mask_sent = [self._tokenizer.mask_token] * len(subwords)
                lm_label_ids = [-1] * len(subwords)
            elif self.masking == "phrase":
                tomask = False
                phrase_rec = []
                for ix, tok in enumerate(subwords):
                    if ix == old2new_first_ixs[start_ix] or tomask:
                        tomask = True
                        mask_sent.append(self._tokenizer.mask_token)
                        lm_label_ids.append(self._tokenizer.convert_tokens_to_ids(tok))
                        phrase_rec += [tok]
                        if phrase_rec == subphrase:
                            tomask = False
                    else:
                        mask_sent.append(tok)
                        lm_label_ids.append(-1)
            elif self.masking == "none":
                mask_sent = subwords
                lm_label_ids = [-1] * len(subwords)

            mask_sent_ids = self._tokenizer.convert_tokens_to_ids(mask_sent)
            mask_sent_ids = mask_sent_ids[: self._max_seq_length - 2]
            mask_sent_ids = self._tokenizer.add_special_tokens_single_sentence(mask_sent_ids)

            lm_label_ids = lm_label_ids[: self._max_seq_length - 2]
            lm_label_ids = [-1] + lm_label_ids + [-1]
            segment_ids = [0] * len(mask_sent_ids)
            input_mask = [1] * len(mask_sent_ids)

            if len(mask_sent_ids) < self._max_seq_length:
                padding = [self._padding_index] * (self._max_seq_length - len(mask_sent_ids))
                mask_sent_ids = mask_sent_ids + padding
                input_mask += padding
                segment_ids += padding
                lm_label_ids += [-1] * len(padding)

            assert_eq(len(mask_sent_ids), self._max_seq_length)
            entry["token"] = mask_sent_ids
            entry["phrase_id"] = phrase_id
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids
            entry["lm_label_ids"] = lm_label_ids

    def tensorize(self):
        for entry in self._entries:
            token = torch.from_numpy(np.array(entry["token"]))
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids

            lm_label_ids = torch.from_numpy(np.array(entry["lm_label_ids"]))
            entry["lm_label_ids"] = lm_label_ids

            phrase_id = torch.from_numpy(np.array(entry["phrase_id"]))
            entry["phrase_id"] = phrase_id

    def __getitem__(self, index):
        entry = self._entries[index]
        image_id = entry["image_id"]
        ref_boxes = entry["refBoxes"]

        features, num_boxes, boxes, boxes_ori, \
            image_cls, obj_labels, obj_confs, attr_labels, attr_confs, image_attrs = self._image_features_reader[image_id]

        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, self._num_locs))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        # mask the overlap regions into zeros
        output_label = np.zeros((mix_features_pad.shape[0])) - 1
        masked_label = np.zeros((mix_features_pad.shape[0]))
        overlaps = iot(torch.tensor(boxes_ori[:, :4]).float(), torch.tensor(ref_boxes).float())
        overlap_threshold = self.threshold
        masked_label = np.logical_or(masked_label, (overlaps > overlap_threshold).max(1)[0])  # overlaps by thr w/ >= 1 ref boxes
        if self._add_global_imgfeat == "first":
            masked_label[0] = 0
        elif self._add_global_imgfeat == "last":
            masked_label[-1] = 0
        # change token to mask token
        mix_features_pad[masked_label > 0] = 0
        # select target region
        iou_overlaps = iou(torch.tensor(boxes_ori[:, :4]).float(), torch.tensor(ref_boxes).float())
        masked_ix = (iou_overlaps.max(1)[0] * masked_label).argmax().item()  # highest IoU over IoT-masked regions
        output_label[masked_ix] = 1

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        caption = entry["token"]
        phrase_id = entry["phrase_id"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]
        lm_label_ids = entry["lm_label_ids"]

        image_cls = image_cls[:mix_num_boxes]
        region_labels = output_label

        return (
            phrase_id,
            caption,
            input_mask,
            segment_ids,
            lm_label_ids,
            features,
            spatials,
            image_cls,
            obj_labels[:mix_num_boxes],
            obj_confs[:mix_num_boxes],
            attr_labels[:mix_num_boxes],
            attr_confs[:mix_num_boxes],
            image_attrs[:mix_num_boxes],
            image_mask,
            region_labels,
            image_id,
        )

    def __len__(self):
        return len(self._entries)
