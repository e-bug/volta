# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import logging
import _pickle as cPickle

import random
import numpy as np

import torch
from torch.utils.data import Dataset

from datasets import load_dataset

logger = logging.getLogger(__name__)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _load_corpus(ann_dir, lgs):
    corpora = {}
    for lg in lgs:
        ann_file = os.path.join(ann_dir, "%s.20180201.txt" % lg)
        corpus = load_dataset('text', split="train", data_files=ann_file, cache_dir=os.path.join(ann_dir, 'huggingface'))
        corpus = corpus.filter(lambda example: len(example['text']) > 0)
        corpora[lg] = corpus
    return corpora


def set_sampling_probs(data, langs, coeff=-1):
    """
    Set the probability of sampling specific languages / language pairs during training.
    """
    if coeff == -1:
        return
    assert coeff > 0
    assert len(langs) > 0

    probs = np.array([1.0 * len(data[lang]) for lang in langs])
    probs /= probs.sum()
    probs = np.array([p ** coeff for p in probs])
    probs /= probs.sum()
    lg2prob = {lg: prob for lg, prob in zip(langs, probs)}
    return lg2prob


def shuf_order(langs, lg2prob, lg_sampling_factor=-1, n=3):
    """
    Randomize training order.
    [https://github.com/microsoft/M3P/blob/master/M3P/src/utils.py]
    """
    if len(langs) == 0:
        return []

    # sample monolingual and parallel languages separately
    mono = langs
    # uniform / weighted sampling
    if lg_sampling_factor == -1:
        p_mono = None
    else:
        p_mono = np.array([lg2prob[k] for k in mono])
        p_mono = p_mono / p_mono.sum()
    s_mono = [mono[i] for i in np.random.choice(len(mono), size=n, p=p_mono, replace=True)]  # min(n, len(mono)), p=p_mono, replace=True)]
    return [lang for lang in s_mono]


class WikipediasDataset(Dataset):
    def __init__(
        self,
        dataroot,
        lgs,
        lg_sampling_factor,
        tokenizer,
        batch_size=512,
        padding_index=0,
        max_seq_length=36,
        max_region_num=36,
        num_locs=5,
        add_global_imgfeat=None,
    ):
        super().__init__()
        self.num_locs = num_locs
        self._max_region_num = max_region_num + int(add_global_imgfeat is not None)
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self._add_global_imgfeat = add_global_imgfeat

        if lgs == ["ALL"]:
            files = glob.glob(os.path.join(dataroot, "*.20180201.txt"))
            self.lgs = []
            for fn in files:
                self.lgs.append(fn.split("/")[-1].split(".")[0])
        else:
            self.lgs = lgs
        self.lg_sampling_factor = lg_sampling_factor
        self.n_sents = batch_size
        self.corpus = _load_corpus(dataroot, self.lgs)
        self.lg2lens = {lang: len(self.corpus[lang]) for lang in self.lgs}
        self.lg2prob = set_sampling_probs(self.corpus, self.lgs, coeff=lg_sampling_factor)

    def random_word(self, tokens, tokenizer):
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability

            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = np.random.randint(len(tokenizer))

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(token)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return tokens, output_label

    def tokenize(self, langs): #, min_seq_len=16):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
#        for ix, line in enumerate(self.corpus):
        entries = []
        for lang in langs:
            ix = np.random.choice(self.lg2lens[lang])
            line = self.corpus[lang][ix]['text']
            # tokenize
            tokens = self._tokenizer.encode(line)

            # add more tokens if len(tokens) < min_len
            _cur = (ix + 1) % len(self.corpus[lang])
            while len(tokens) < self._max_seq_length-2:
                _cur_tokens = self._tokenizer.encode(self.corpus[lang][_cur]['text'])
                tokens += _cur_tokens
                _cur = (_cur + 1) % len(self.corpus[lang])

            # truncate
            tokens = tokens[:self._max_seq_length - 2]
            tokens, tokens_label = self.random_word(tokens, self._tokenizer)
            tokens = self._tokenizer.build_inputs_with_special_tokens(tokens)
            lm_label_ids = [-1] + tokens_label + [-1]

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            assert_eq(len(tokens), self._max_seq_length)
            entry = {
                "q_token": tokens,
                "q_input_mask": input_mask,
                "q_segment_ids": segment_ids,
                "q_label": lm_label_ids,
            }
            entries.append(entry)
        return entries

    def tensorize(self, entries):
        for entry in entries:
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

            q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
            entry["q_input_mask"] = q_input_mask

            q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
            entry["q_segment_ids"] = q_segment_ids

            q_label_ids = torch.from_numpy(np.array(entry["q_label"]))
            entry["q_label"] = q_label_ids

    def __getitem__(self, index=0):
        # Image
        max_region_num = self._max_region_num + int(self._add_global_imgfeat is not None)
        features = torch.zeros((self.n_sents, max_region_num, 2048), dtype=torch.float)
        spatials = torch.zeros((self.n_sents, max_region_num, self.num_locs), dtype=torch.float)
        image_mask = torch.zeros((self.n_sents, max_region_num), dtype=torch.long)

        # Text
        langs = shuf_order(self.lgs, self.lg2prob, self.lg_sampling_factor, n=self.n_sents)
        entries = self.tokenize(langs)
        self.tensorize(entries)
        input_ids = torch.stack([entry["q_token"] for entry in entries])
        input_mask = torch.stack([entry["q_input_mask"] for entry in entries])
        segment_ids = torch.stack([entry["q_segment_ids"] for entry in entries])
        lm_label_ids = torch.stack([entry["q_label"] for entry in entries])

        return input_ids, input_mask, segment_ids, lm_label_ids, features, spatials, image_mask

    def sample(self):
        return self.__getitem__()

    def __len__(self):
        return len(self.corpus)
