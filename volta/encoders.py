# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import math
import logging

import torch
from torch import nn
import torch.nn.functional as F

from .embeddings import *
from .config import BertConfig
from .utils import PreTrainedModel
from .losses import pre_vis_criterions, pre_vis_targets


logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    "bert-base-german-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
}

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")

    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                l = re.split(r"_(\d+)", m_name)
            else:
                l = [m_name]
            if l[0] == "kernel" or l[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif l[0] == "output_bias" or l[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif l[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


# ==================================================================================================================== #
#                                                 Activation Functions                                                 #
# ==================================================================================================================== #
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


# ==================================================================================================================== #
#                                                     Gated layers                                                     #
# ==================================================================================================================== #
class BertGatedSelfAttention(nn.Module):
    def __init__(self, config, layer_num):
        super(BertGatedSelfAttention, self).__init__()

        hidden_size = config.sublayer2attn_hidden_size.get(str(layer_num), config.hidden_size)
        num_attention_heads = config.sublayer2num_attention_heads.get(str(layer_num), config.num_attention_heads)
        v_hidden_size = config.sublayer2v_attn_hidden_size.get(str(layer_num), config.v_hidden_size)
        v_num_attention_heads = config.sublayer2v_num_attention_heads.get(str(layer_num), config.v_num_attention_heads)

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The text hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if v_hidden_size % v_num_attention_heads != 0:
            raise ValueError(
                "The vision hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (v_hidden_size, num_attention_heads)
            )
        self.v_num_attention_heads = v_num_attention_heads
        self.v_attention_head_size = int(v_hidden_size / v_num_attention_heads)
        self.v_all_head_size = self.v_num_attention_heads * self.v_attention_head_size

        self.visualization = config.visualization

        self.has_tt = (layer_num in config.tt_attn_sublayers)
        self.has_tv = (layer_num in config.tv_attn_sublayers)
        self.has_vt = (layer_num in config.vt_attn_sublayers)
        self.has_vv = (layer_num in config.vv_attn_sublayers)
        self.has_text = (self.has_tt or self.has_tv)
        self.has_vision = (self.has_vv or self.has_vt)
        self.share_layer = (layer_num in config.shared_sublayers)
        if self.has_tv or self.has_vt:
            assert hidden_size == v_hidden_size, "hidden_size != v_hidden_size"
            assert num_attention_heads == v_num_attention_heads, "num_attention_heads != v_num_attention_heads"

        if self.has_text:
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        if self.has_text and self.has_vision and self.share_layer:
            assert hidden_size == v_hidden_size, "hidden_size != v_hidden_size"
            self.v_query = self.query
            self.v_key = self.key
            self.v_value = self.value
            self.v_dropout = self.dropout
        elif self.has_vision:
            self.v_query = nn.Linear(config.v_hidden_size, self.v_all_head_size)
            self.v_key = nn.Linear(config.v_hidden_size, self.v_all_head_size)
            self.v_value = nn.Linear(config.v_hidden_size, self.v_all_head_size)
            self.v_dropout = nn.Dropout(config.v_attention_probs_dropout_prob)

    def transpose_for_scores(self, x, modality='text'):
        if modality == 'text':
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        else:
            new_x_shape = x.size()[:-1] + (self.v_num_attention_heads, self.v_attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, t_hidden_states, v_hidden_states, t_attention_mask, v_attention_mask):
        """
        Args:
            t_hidden_states: [bs, seq_len, hidden_size]
            v_hidden_states: [bs, num_box, v_hidden_size]
            t_attention_mask: [bs, 1, 1, seq_len] filled with 0s (non-pad) and -10000s (pad)
            v_attention_mask: [bs, 1, 1, num_box] filled with 0s (non-pad) and -10000s (pad)
        Returns:
            t_context_layer: [bs, seq_len, hidden_size] or int 0 if no text
            v_context_layer: [bs, num_box, v_hidden_size] or int 0 if no vision
            t_attn_data: dict or None if no visualization
            v_attn_data: dict or None if no visualization
        """
        if self.has_text:
            t_mixed_query_layer = self.query(t_hidden_states)  # [bs, seq_len, hidden_size]
            t_mixed_key_layer = self.key(t_hidden_states)
            t_mixed_value_layer = self.value(t_hidden_states)
            t_query_layer = self.transpose_for_scores(t_mixed_query_layer)  # [bs, num_heads, seq_len, attn_head_size]
            t_key_layer = self.transpose_for_scores(t_mixed_key_layer)
            t_value_layer = self.transpose_for_scores(t_mixed_value_layer)

        if self.has_vision:
            v_mixed_query_layer = self.v_query(v_hidden_states)  # [bs, num_box, v_hidden_size]
            v_mixed_key_layer = self.v_key(v_hidden_states)
            v_mixed_value_layer = self.v_value(v_hidden_states)
            v_query_layer = self.transpose_for_scores(v_mixed_query_layer, 'vision')  # [bs, v_num_heads, num_box, v_attn_head_size]
            v_key_layer = self.transpose_for_scores(v_mixed_key_layer, 'vision')
            v_value_layer = self.transpose_for_scores(v_mixed_value_layer, 'vision')

        # Gated attention
        if self.has_tt:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            tt_attention_scores = torch.matmul(t_query_layer, t_key_layer.transpose(-1, -2))  # [bs, num_heads, seq_len, seq_len]
            tt_attention_scores = tt_attention_scores / math.sqrt(self.attention_head_size)
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            tt_attention_scores = tt_attention_scores + t_attention_mask  # [bs, num_heads, seq_len, seq_len]
        if self.has_tv:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            tv_attention_scores = torch.matmul(t_query_layer, v_key_layer.transpose(-1, -2))  # [bs, num_heads, seq_len, num_box]
            tv_attention_scores = tv_attention_scores / math.sqrt(self.attention_head_size)
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            tv_attention_scores = tv_attention_scores + v_attention_mask
        if self.has_vt:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            vt_attention_scores = torch.matmul(v_query_layer, t_key_layer.transpose(-1, -2))  # [bs, num_heads, num_box, seq_len]
            vt_attention_scores = vt_attention_scores / math.sqrt(self.v_attention_head_size)
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            vt_attention_scores = vt_attention_scores + t_attention_mask
        if self.has_vv:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            vv_attention_scores = torch.matmul(v_query_layer, v_key_layer.transpose(-1, -2))  # [bs, num_heads, num_box, num_box]
            vv_attention_scores = vv_attention_scores / math.sqrt(self.v_attention_head_size)
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            vv_attention_scores = vv_attention_scores + v_attention_mask

        # Gated softmax
        # Normalize the attention scores to probabilities.
        if self.has_tt and self.has_tv:
            # Concatenate the two attention scores
            t_attention_scores = torch.cat((tt_attention_scores, tv_attention_scores), dim=-1)  # [bs, num_heads, seq_len, seq_len+num_box]
            t_attention_probs = nn.Softmax(dim=-1)(t_attention_scores)
            # Split concatenation back into tt and tv
            tt_attention_probs, tv_attention_probs = \
                t_attention_probs.split([tt_attention_scores.size(-1), tv_attention_scores.size(-1)], dim=-1)
            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            tt_attention_probs = self.dropout(tt_attention_probs)  # [bs, num_heads, seq_len, seq_len]
            tv_attention_probs = self.dropout(tv_attention_probs)  # [bs, num_heads, seq_len, num_box]
        elif self.has_tt:
            tt_attention_probs = self.dropout(nn.Softmax(dim=-1)(tt_attention_scores))  # [bs, num_heads, seq_len, seq_len]
        elif self.has_tv:
            tv_attention_probs = self.dropout(nn.Softmax(dim=-1)(tv_attention_scores))  # [bs, num_heads, seq_len, num_box]
        if self.has_vv and self.has_vt:
            # Concatenate the two attention scores
            v_attention_scores = torch.cat((vt_attention_scores, vv_attention_scores), dim=-1)  # [bs, num_heads, seq_len, seq_len+num_box]
            v_attention_probs = nn.Softmax(dim=-1)(v_attention_scores)
            # Split concatenation back into vt and vv
            vt_attention_probs, vv_attention_probs = \
                v_attention_probs.split([vt_attention_scores.size(-1), vv_attention_scores.size(-1)], dim=-1)
            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            vv_attention_probs = self.v_dropout(vv_attention_probs)  # [bs, num_heads, num_box, num_box]
            vt_attention_probs = self.v_dropout(vt_attention_probs)  # [bs, num_heads, num_box, seq_len]
        elif self.has_vv:
            vv_attention_probs = self.v_dropout(nn.Softmax(dim=-1)(vv_attention_scores))  # [bs, v_num_heads, num_box, num_box]
        elif self.has_vt:
            vt_attention_probs = self.v_dropout(nn.Softmax(dim=-1)(vt_attention_scores))  # [bs, num_heads, num_box, seq_len]

        # Gated context
        tt_context_layer, tv_context_layer, vt_context_layer, vv_context_layer = 0, 0, 0, 0
        if self.has_tt:
            tt_context_layer = torch.matmul(tt_attention_probs, t_value_layer)  # [bs, num_heads, seq_len, attn_head_size]
            tt_context_layer = tt_context_layer.permute(0, 2, 1, 3).contiguous()  # [bs, seq_len, num_heads, attn_head_size]
            tt_new_context_layer_shape = tt_context_layer.size()[:-2] + (self.all_head_size,)
            tt_context_layer = tt_context_layer.view(*tt_new_context_layer_shape)  # [bs, seq_len, num_heads*attn_head_size]
        if self.has_tv:
            tv_context_layer = torch.matmul(tv_attention_probs, v_value_layer)
            tv_context_layer = tv_context_layer.permute(0, 2, 1, 3).contiguous()
            tv_new_context_layer_shape = tv_context_layer.size()[:-2] + (self.all_head_size,)
            tv_context_layer = tv_context_layer.view(*tv_new_context_layer_shape)
        if self.has_vt:
            vt_context_layer = torch.matmul(vt_attention_probs, t_value_layer)
            vt_context_layer = vt_context_layer.permute(0, 2, 1, 3).contiguous()
            vt_new_context_layer_shape = vt_context_layer.size()[:-2] + (self.v_all_head_size,)
            vt_context_layer = vt_context_layer.view(*vt_new_context_layer_shape)
        if self.has_vv:
            vv_context_layer = torch.matmul(vv_attention_probs, v_value_layer)  # [bs, v_num_heads, num_box, v_attn_head_size]
            vv_context_layer = vv_context_layer.permute(0, 2, 1, 3).contiguous()  # [bs, num_box, v_num_heads, v_attn_head_size]
            vv_new_context_layer_shape = vv_context_layer.size()[:-2] + (self.v_all_head_size,)
            vv_context_layer = vv_context_layer.view(*vv_new_context_layer_shape)  # [bs, num_box, v_num_heads*v_attn_head_size]

        t_context_layer = (tt_context_layer + tv_context_layer)  # [bs, seq_len, hidden_size] or int 0 if no text
        v_context_layer = (vv_context_layer + vt_context_layer)  # [bs, num_box, v_hidden_size] or int 0 if no vision

        if self.visualization:
            t_attn_data = {
                "intra_attn": tt_attention_probs if self.has_tt else None,
                "inter_attn": tv_attention_probs if self.has_tv else None,
                "queries": t_query_layer if self.has_text else None,
                "keys": t_key_layer if self.has_text else None,
            }
            v_attn_data = {
                "intra_attn": vv_attention_probs if self.has_vv else None,
                "inter_attn": vt_attention_probs if self.has_vt else None,
                "queries": v_query_layer if self.has_vision else None,
                "keys": v_key_layer if self.has_vision else None,
            }
        else:
            t_attn_data, v_attn_data = None, None

        return t_context_layer, v_context_layer, t_attn_data, v_attn_data


class BertGatedSelfOutput(nn.Module):
    def __init__(self, config, layer_num):
        super(BertGatedSelfOutput, self).__init__()

        hidden_size = config.sublayer2attn_hidden_size.get(str(layer_num), config.hidden_size)
        v_hidden_size = config.sublayer2v_attn_hidden_size.get(str(layer_num), config.v_hidden_size)

        self.has_language = ((layer_num in config.tt_attn_sublayers) or (layer_num in config.tv_attn_sublayers))
        self.has_vision = ((layer_num in config.vv_attn_sublayers) or (layer_num in config.vt_attn_sublayers))
        self.share_layer = (layer_num in config.shared_sublayers)
        self.single_ln = (layer_num in config.single_ln_sublayers)
        if self.single_ln:
            assert (self.has_language and self.has_vision and self.share_layer), "Missing language, vision or sharing"

        if self.has_language:
            self.dense = nn.Linear(hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        else:
            self.dense = lambda x: x
            self.dropout = lambda x: x
            self.LayerNorm = lambda x: x

        if self.has_language and self.has_vision and self.share_layer:
            assert (hidden_size == v_hidden_size) and (config.hidden_size == config.v_hidden_size), "hidden_size != v_hidden_size"
            self.v_dense = self.dense
            self.v_dropout = self.dropout
            self.v_LayerNorm = self.LayerNorm
        elif self.has_vision:
            self.v_dense = nn.Linear(v_hidden_size, config.v_hidden_size)
            self.v_dropout = nn.Dropout(config.v_hidden_dropout_prob)
            self.v_LayerNorm = BertLayerNorm(v_hidden_size, eps=1e-12)
        else:
            self.v_dense = lambda x: x
            self.v_dropout = lambda x: x
            self.v_LayerNorm = lambda x: x

    def forward(self, t_hidden_states, v_hidden_states, t_input_tensor, v_input_tensor):
        """
        Args:
            t_hidden_states: [bs, seq_len, hidden_size] or int 0 if no text
            v_hidden_states: [bs, num_box, v_hidden_size] or int 0 if no vision
            t_input_tensor: [bs, seq_len, hidden_size]
            v_input_tensor: [bs, num_box, v_hidden_size]
        Returns:
            t_hidden_states: [bs, seq_len, hidden_size]
            v_hidden_states: [bs, num_box, v_hidden_size]
        """

        t_hidden_states = self.dense(t_hidden_states)
        t_hidden_states = self.dropout(t_hidden_states)
        v_hidden_states = self.v_dense(v_hidden_states)
        v_hidden_states = self.v_dropout(v_hidden_states)
        if self.single_ln:
            # Concatenate text and vision
            hidden_states = torch.cat((t_hidden_states, v_hidden_states), dim=1)  # [bs, seq_len+num_box, hidden_size]
            inputs = torch.cat((t_input_tensor, v_input_tensor), dim=1)  # [bs, seq_len+num_box, hidden_size]
            hidden_states = self.LayerNorm(hidden_states + inputs)
            t_hidden_states, v_hidden_states = \
                hidden_states.split([t_hidden_states.size(1), v_hidden_states.size(1)], dim=1)
        else:
            t_hidden_states = self.LayerNorm(t_hidden_states + t_input_tensor)
            v_hidden_states = self.v_LayerNorm(v_hidden_states + v_input_tensor)
        return t_hidden_states, v_hidden_states


class BertGatedAttention(nn.Module):
    def __init__(self, config, layer_num):
        super(BertGatedAttention, self).__init__()
        self.attention_self = BertGatedSelfAttention(config, layer_num)
        self.attention_output = BertGatedSelfOutput(config, layer_num)

    def forward(self, t_input_tensor, v_input_tensor, t_attention_mask, v_attention_mask):
        """
        Args:
            t_input_tensor: [bs, seq_len, hidden_size]
            v_input_tensor: [bs, num_box, v_hidden_size]
            t_attention_mask: [bs, 1, 1, seq_len] filled with 0s (non-pad) and -10000s (pad)
            v_attention_mask: [bs, 1, 1, num_box] filled with 0s (non-pad) and -10000s (pad)
        Returns:
            t_attn_output: [bs, seq_len, hidden_size]
            v_attn_output: [bs, num_box, v_hidden_size]
            t_attn_probs: dict or None if no visualization
            v_attn_probs: dict or None if no visualization
        """
        t_self_output, v_self_output, t_attn_probs, v_attn_probs = self.attention_self(t_input_tensor, v_input_tensor,
                                                                                       t_attention_mask, v_attention_mask)
        t_attn_output, v_attn_output = self.attention_output(t_self_output, v_self_output, t_input_tensor, v_input_tensor)
        return t_attn_output, v_attn_output, t_attn_probs, v_attn_probs


class BertGatedIntermediate(nn.Module):
    def __init__(self, config, layer_num):
        super(BertGatedIntermediate, self).__init__()
        self.has_language = (layer_num in config.t_ff_sublayers)
        self.has_vision = (layer_num in config.v_ff_sublayers)
        self.share_layer = (layer_num in config.shared_sublayers)

        intermediate_size = config.sublayer2intermediate_size.get(str(layer_num), config.intermediate_size)
        v_intermediate_size = config.sublayer2v_intermediate_size.get(str(layer_num), config.v_intermediate_size)

        if self.has_language:
            self.dense = nn.Linear(config.hidden_size, intermediate_size)
            if isinstance(config.hidden_act, str):
                self.intermediate_act_fn = ACT2FN[config.hidden_act]
            else:
                self.intermediate_act_fn = config.hidden_act
        else:
            self.dense = lambda x: x
            self.intermediate_act_fn = lambda x: 0
        if self.has_language and self.has_vision and self.share_layer:
            assert config.hidden_size == config.v_hidden_size, "hidden_size != v_hidden_size"
            assert intermediate_size == v_intermediate_size, "intermediate_size != v_intermediate_size"
            self.v_dense = self.dense
            self.v_intermediate_act_fn = self.intermediate_act_fn
        elif self.has_vision:
            self.v_dense = nn.Linear(config.v_hidden_size, v_intermediate_size)
            if isinstance(config.hidden_act, str):
                self.v_intermediate_act_fn = ACT2FN[config.v_hidden_act]
            else:
                self.v_intermediate_act_fn = config.v_hidden_act
        else:
            self.v_dense = lambda x: x
            self.v_intermediate_act_fn = lambda x: 0

    def forward(self, t_hidden_states, v_hidden_states):
        """
        Args:
            t_hidden_states: [bs, seq_len, hidden_size]
            v_hidden_states: [bs, num_box, v_hidden_size]
        Returns:
            t_hidden_states: [bs, seq_len, hidden_size] or int 0 if no text
            v_hidden_states: [bs, num_box, v_hidden_size] or int 0 if no vision
        """
        t_hidden_states = self.dense(t_hidden_states)
        t_hidden_states = self.intermediate_act_fn(t_hidden_states)

        v_hidden_states = self.v_dense(v_hidden_states)
        v_hidden_states = self.v_intermediate_act_fn(v_hidden_states)

        return t_hidden_states, v_hidden_states


class BertGatedOutput(nn.Module):
    def __init__(self, config, layer_num):
        super(BertGatedOutput, self).__init__()
        self.has_language = (layer_num in config.t_ff_sublayers)
        self.has_vision = (layer_num in config.v_ff_sublayers)
        self.share_layer = (layer_num in config.shared_sublayers)
        self.single_ln = (layer_num in config.single_ln_sublayers)
        if self.single_ln:
            assert (self.has_language and self.has_vision and self.share_layer), "Missing language, vision or sharing"

        intermediate_size = config.sublayer2intermediate_size.get(str(layer_num), config.intermediate_size)
        v_intermediate_size = config.sublayer2v_intermediate_size.get(str(layer_num), config.v_intermediate_size)

        if self.has_language:
            self.dense = nn.Linear(intermediate_size, config.hidden_size)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        else:
            self.dense = lambda x: x
            self.dropout = lambda x: x
            self.LayerNorm = lambda x: x

        if self.has_language and self.has_vision and self.share_layer:
            assert config.hidden_size == config.v_hidden_size, "hidden_size != v_hidden_size"
            assert intermediate_size == v_intermediate_size, "intermediate_size != v_intermediate_size"
            self.v_dense = self.dense
            self.v_dropout = self.dropout
            self.v_LayerNorm = self.LayerNorm
        elif self.has_vision:
            self.v_dense = nn.Linear(v_intermediate_size, config.v_hidden_size)
            self.v_dropout = nn.Dropout(config.v_hidden_dropout_prob)
            self.v_LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        else:
            self.v_dense = lambda x: x
            self.v_dropout = lambda x: x
            self.v_LayerNorm = lambda x: x

    def forward(self, t_hidden_states, v_hidden_states, t_input_tensor, v_input_tensor):
        """
        Args:
            t_hidden_states: [bs, seq_len, hidden_size] or int 0 if no text
            v_hidden_states: [bs, num_box, v_hidden_size] or int 0 if no vision
            t_input_tensor: [bs, seq_len, hidden_size]
            v_input_tensor: [bs, num_box, v_hidden_size]
        Returns:
            t_hidden_states: [bs, seq_len, hidden_size]
            v_hidden_states: [bs, num_box, v_hidden_size]
        """
        t_hidden_states = self.dense(t_hidden_states)
        t_hidden_states = self.dropout(t_hidden_states)
        v_hidden_states = self.v_dense(v_hidden_states)
        v_hidden_states = self.v_dropout(v_hidden_states)
        if self.single_ln:
            # Concatenate text and vision
            hidden_states = torch.cat((t_hidden_states, v_hidden_states), dim=1)  # [bs, seq_len+num_box, hidden_size]
            inputs = torch.cat((t_input_tensor, v_input_tensor), dim=1)  # [bs, seq_len+num_box, hidden_size]
            hidden_states = self.LayerNorm(hidden_states + inputs)
            t_hidden_states, v_hidden_states = \
                hidden_states.split([t_hidden_states.size(1), v_hidden_states.size(1)], dim=1)
        else:
            t_hidden_states = self.LayerNorm(t_hidden_states + t_input_tensor)
            v_hidden_states = self.v_LayerNorm(v_hidden_states + v_input_tensor)
        return t_hidden_states, v_hidden_states


class BertGatedFeedForward(nn.Module):
    def __init__(self, config, layer_num):
        super(BertGatedFeedForward, self).__init__()
        self.intermediate = BertGatedIntermediate(config, layer_num)
        self.output = BertGatedOutput(config, layer_num)

    def forward(self, t_input_tensor, v_input_tensor):
        """
        Args:
            t_input_tensor: [bs, seq_len, hidden_size]
            v_input_tensor: [bs, num_box, v_hidden_size]
            # t_attention_probs: dict or None if no visualization
            # v_attention_probs: dict or None if no visualization
        Returns:
            t_layer_output: [bs, seq_len, hidden_size]
            v_layer_output: [bs, num_box, v_hidden_size]
            # t_attention_probs: dict or None if no visualization
            # v_attention_probs: dict or None if no visualization
        """
        t_inter_output, v_inter_output = self.intermediate(t_input_tensor, v_input_tensor)
        t_layer_output, v_layer_output = self.output(t_inter_output, v_inter_output, t_input_tensor, v_input_tensor)
        return t_layer_output, v_layer_output


# ==================================================================================================================== #
#                                                       Pooling                                                        #
# ==================================================================================================================== #
class BertTextPooler(nn.Module):
    def __init__(self, config):
        super(BertTextPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.pooler_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class VLBertTextPooler(nn.Module):
    def __init__(self, config):
        super(VLBertTextPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.pooler_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states, text_end):
        # We "pool" the model by simply taking the hidden state corresponding to the first token.
        _, grid_pos = torch.meshgrid(torch.arange(hidden_states.size(0), dtype=torch.long, device=hidden_states.device),
                                     torch.arange(hidden_states.size(1), dtype=torch.long, device=hidden_states.device))
        mask_token_tensor = hidden_states[(grid_pos == text_end - 2)]
        pooled_output = self.dense(mask_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertImagePooler(nn.Module):
    def __init__(self, config):
        super(BertImagePooler, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_pooler_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# ==================================================================================================================== #
#                                                        Heads                                                         #
# ==================================================================================================================== #
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertImgPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertImgPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.v_hidden_act
        if config.image_head_ln:
            self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        else:
            self.LayerNorm = lambda x: x

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertImagePredictionHead(nn.Module):
    def __init__(self, config):
        super(BertImagePredictionHead, self).__init__()
        self.transform = BertImgPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is an output-only bias for each token.
        self.decoder_dict = nn.ModuleDict({
            ix: nn.Linear(config.v_hidden_size, num)
            for ix, num in pre_vis_targets.items()
            if config.visual_target_weights.get(ix, 0) > 0
        })

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = {}
        for ix in self.decoder_dict:
            output[ix] = self.decoder_dict[ix](hidden_states)
        return output


class LxmertAnswerHead(nn.Module):
    def __init__(self, config, num_answers):
        super().__init__()
        hid_dim = config.v_hidden_size
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        if config.fusion_method in {"none", "vl-bert_vqa"}:
            self.bi_seq_relationship = lambda x: None
        else:
            self.bi_seq_relationship = nn.Linear(config.pooler_size, 2)
        self.imagePredictions = BertImagePredictionHead(config)
        if config.qa_task_weight:
            self.qaPredictions = LxmertAnswerHead(config)
        else:
            self.qaPredictions = lambda x: None
        self.fusion_method = config.fusion_method
        self.dropout = nn.Dropout(0.1)
        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v):
        if self.fusion_method == "sum":
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == "mul":
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        elif self.fusion_method == "text":
            pooled_output = self.dropout(pooled_output_t)
        elif self.fusion_method == "vl-bert_vqa":
            pooled_output = self.dropout(pooled_output_t)
        elif self.fusion_method == "none":
            pooled_output = None
        else:
            assert False

        prediction_scores_t = self.predictions(sequence_output_t)
        seq_relationship_score = self.bi_seq_relationship(pooled_output)
        prediction_scores_v_dict = self.imagePredictions(sequence_output_v)
        vqa_score = self.qaPredictions(pooled_output)

        return prediction_scores_t, prediction_scores_v_dict, seq_relationship_score, vqa_score, pooled_output


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout_prob=0.0):
        super().__init__()

        self.logit_fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            GeLU(),
            BertLayerNorm(hid_dim, eps=1e-12),
            nn.Linear(hid_dim, out_dim),
        )

        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)


# ==================================================================================================================== #
#                                                        Models                                                        #
# ==================================================================================================================== #
class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()

        # in the bert encoder, we need to extract three things here.
        # text bert layer: BertLayer
        # vision bert layer: BertImageLayer
        # Bi-Attention: Given the output of two bertlayer, perform bi-directional
        # attention and add on two layers.

        attn_sublayers = set(config.tt_attn_sublayers + config.tv_attn_sublayers +
                             config.vt_attn_sublayers + config.vv_attn_sublayers)
        ff_sublayers = set(config.t_ff_sublayers + config.v_ff_sublayers)
        depth = len(attn_sublayers) + len(ff_sublayers)
        num2layers = {}
        self.num2type = {}
        for n in attn_sublayers:
            num2layers[n] = BertGatedAttention(config, n)
            self.num2type[n] = "attn"
        for n in ff_sublayers:
            num2layers[n] = BertGatedFeedForward(config, n)
            self.num2type[n] = "ff"
        assert len(num2layers) == depth, "Overlapping attn-ff sublayer numbers"
        assert (min(num2layers) == 0) and (max(num2layers) == depth - 1), "Non contiguous sublayer numbers"

        self.layer = nn.ModuleList([copy.deepcopy(sublayer) for _, sublayer in sorted(num2layers.items())])

    def forward(self, txt_embedding, img_embedding, txt_attention_mask, img_attention_mask,
                output_all_encoded_layers=True, output_all_attention_masks=False):
        """
        Args:
            txt_embedding: [bs, seq_len, hidden_size]
            img_embedding: [bs, num_box, v_hidden_size]
            txt_attention_mask: [bs, 1, 1, seq_len] filled with 0s (non-pad) and -10000s (pad)
            img_attention_mask: [bs, 1, 1, num_box] filled with 0s (non-pad) and -10000s (pad)
            output_all_encoded_layers: Bool
            output_all_attention_masks: Bool
        Returns:
            all_encoder_layers_t:
            all_encoder_layers_v:
            (all_attention_mask_t, all_attention_mask_v):
        """
        all_encoder_layers_t = [txt_embedding]
        all_encoder_layers_v = [img_embedding]

        all_attention_mask_t = [None]
        all_attention_mask_v = [None]

        for idx, layer in enumerate(self.layer):
            layer_type = self.num2type[idx]
            if layer_type == "attn":
                txt_embedding, img_embedding, txt_attention_probs, img_attention_probs = \
                    layer(txt_embedding, img_embedding, txt_attention_mask, img_attention_mask)
                if output_all_attention_masks:
                    all_attention_mask_t.append(txt_attention_probs)
                    all_attention_mask_v.append(img_attention_probs)
            else:
                txt_embedding, img_embedding = layer(txt_embedding, img_embedding)
                if output_all_attention_masks:
                    all_attention_mask_t.append(None)
                    all_attention_mask_v.append(None)

            if output_all_encoded_layers:
                all_encoder_layers_t.append(txt_embedding)
                all_encoder_layers_v.append(img_embedding)

        # add the end part to finish.
        if not output_all_encoded_layers:
            all_encoder_layers_t.append(txt_embedding)
            all_encoder_layers_v.append(img_embedding)

        return all_encoder_layers_t, all_encoder_layers_v, (all_attention_mask_t, all_attention_mask_v)


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def __init__(self, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__(*inputs, **kwargs)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.shared_embeddings = False

        # initialize word embedding
        if "roberta" in config.bert_model:
            self.embeddings = RobertaEmbeddings(config)
        elif "bert" in config.bert_model:
            self.embeddings = BertEmbeddings(config)

        # initialize vision embedding
        if config.image_embeddings in dual_embeddings:
            self.v_embeddings = dual_embeddings[config.image_embeddings](config)
        else:
            self.v_embeddings = lambda x, y: None

        self.encoder = BertEncoder(config)
        self.fusion_method = config.fusion_method
        if config.fusion_method == "none":
            self.t_pooler = lambda x: None
        elif config.fusion_method == "vl-bert_vqa":
            self.t_pooler = VLBertTextPooler(config)
        else:
            self.t_pooler = BertTextPooler(config)
        if config.fusion_method in {"none", "text", "vl-bert_vqa"}:
            self.v_pooler = lambda x: None
        else:
            assert config.pooler_size == config.v_pooler_size, "pooler_size != v_pooler_size"
            self.v_pooler = BertImagePooler(config)

        self.apply(self.init_weights)
        if config.image_embeddings in shared_embeddings:
            self.embeddings = shared_embeddings[config.image_embeddings](config)
            self.shared_embeddings = True

    def forward(self, input_txt, input_imgs, image_loc, token_type_ids=None, attention_mask=None,
                image_attention_mask=None, output_all_encoded_layers=False, output_all_attention_masks=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_txt)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_txt)
        if image_attention_mask is None:
            image_attention_mask = torch.ones(input_imgs.size(0), input_imgs.size(1)).type_as(input_txt)

        if self.shared_embeddings:
            embedding_output, v_embedding_output = self.embeddings(input_txt, input_imgs, image_loc, token_type_ids)
        else:
            embedding_output = self.embeddings(input_txt, token_type_ids)
            v_embedding_output = self.v_embeddings(input_imgs, image_loc)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = image_attention_mask.unsqueeze(1).unsqueeze(2)

        # extended_attention_mask2 = attention_mask.unsqueeze(2)
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        extended_image_attention_mask = extended_image_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0

        encoded_layers_t, encoded_layers_v, all_attention_mask = self.encoder(
            embedding_output,
            v_embedding_output,
            extended_attention_mask,
            extended_image_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            output_all_attention_masks=output_all_attention_masks,
        )

        sequence_output_t = encoded_layers_t[-1]
        sequence_output_v = encoded_layers_v[-1]

        if self.fusion_method == "vl-bert_vqa":
            text_mask = input_txt != 0
            text_end = text_mask.sum(1, keepdim=True)
            pooled_output_t = self.t_pooler(sequence_output_t, text_end)
        else:
            pooled_output_t = self.t_pooler(sequence_output_t)
        pooled_output_v = self.v_pooler(sequence_output_v)

        if not output_all_encoded_layers:
            encoded_layers_t = encoded_layers_t[-1]
            encoded_layers_v = encoded_layers_v[-1]

        return encoded_layers_t, encoded_layers_v, pooled_output_t, pooled_output_v, all_attention_mask


class BertForVLPreTraining(BertPreTrainedModel):
    """BERT model with multimodal pre-training heads.
    """

    def __init__(self, config):
        super(BertForVLPreTraining, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.visual_target_weights = config.visual_target_weights
        self.qa_weight = config.qa_task_weight
        self.qa_num_answers = config.qa_num_answers
        print("model's visual targets are ", [ix for ix, w in config.visual_target_weights.items() if w > 0])
        if self.qa_weight:
            print("model's qa target is ", self.qa_weight)

        self.add_global_imgfeat = int(config.add_global_imgfeat is not None)

        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder, self.bert.embeddings.word_embeddings)

    def forward(
        self,
        input_ids,
        image_feat,
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        masked_lm_labels=None,
        image_label=None,
        image_cls=None,
        obj_labels=None,
        obj_confs=None,
        attr_labels=None,
        attr_confs=None,
        image_attrs=None,
        next_sentence_label=None,
        answers=None,
        output_all_encoded_layers=False,
        output_all_attention_masks=False,
    ):
        # in this model, we first embed the images.
        encoded_layers_t, encoded_layers_v, pooled_output_t, pooled_output_v, all_attention_mask = self.bert(
            input_ids,
            image_feat,
            image_loc,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            output_all_attention_masks=output_all_attention_masks,
        )
        if output_all_encoded_layers:
            sequence_output_t = encoded_layers_t[-1]
            sequence_output_v = encoded_layers_v[-1]
        else:
            sequence_output_t = encoded_layers_t
            sequence_output_v = encoded_layers_v

        prediction_scores_t, prediction_scores_v_dict, seq_relationship_score, vqa_score, pooled_output = self.cls(
            sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v
        )

        # Vision loss
        img_loss = 0
        for ix, weight in self.visual_target_weights.items():
            if self.config.add_global_imgfeat == "last":
                prediction_scores_v = prediction_scores_v_dict[ix][:, :-1]
            else:
                prediction_scores_v = prediction_scores_v_dict[ix][:, self.add_global_imgfeat:]
            img_loss += pre_vis_criterions[ix](prediction_scores_v, weight, image_label, image_cls, image_feat,
                                               obj_labels, obj_confs, attr_labels, attr_confs)
        if self.qa_weight and answers:
            img_loss += self.qa_weight * self.loss_fct(vqa_score.view(-1, self.qa_num_answers), answers.view(-1))

        masked_img_loss = img_loss > 0 if type(img_loss) == int else img_loss.cpu().item() > 0
        if masked_img_loss:
            img_loss = img_loss.unsqueeze(0)
        else:
            img_loss = torch.zeros(1).to(prediction_scores_t.device)

        if masked_lm_labels is not None:
            masked_lm_loss = self.loss_fct(
                prediction_scores_t.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
            ).unsqueeze(0)
        else:
            masked_lm_loss = torch.zeros(1).to(prediction_scores_t.device)

        if (seq_relationship_score is not None) and (next_sentence_label is not None):
            next_sentence_loss = self.loss_fct(
                seq_relationship_score.view(-1, 2),
                next_sentence_label.view(-1)
            ).unsqueeze(0)
        else:
            next_sentence_loss = torch.zeros(1).to(prediction_scores_t.device)

        if masked_img_loss or masked_lm_loss or next_sentence_loss:
            if output_all_encoded_layers:
                return masked_lm_loss, img_loss, next_sentence_loss, encoded_layers_t, encoded_layers_v
            return masked_lm_loss, img_loss, next_sentence_loss
        else:
            if output_all_encoded_layers:
                return prediction_scores_t, prediction_scores_v_dict, seq_relationship_score, all_attention_mask, \
                        pooled_output, encoded_layers_t, encoded_layers_v
            return prediction_scores_t, prediction_scores_v_dict, seq_relationship_score, vqa_score, \
                    all_attention_mask, pooled_output


class BertForVLTasks(BertPreTrainedModel):
    def __init__(self, config, task_cfg, task_ids, dropout_prob=0.1):
        super(BertForVLTasks, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout_prob)

        self.config = config
        self.task_cfg = task_cfg
        task2clf = {}
        for task_id in task_ids:
            task_type = task_cfg[task_id]["type"]
            if task_type in {"VL-classifier", "VL-classifier-GQA"}:
                task2clf[task_id] = SimpleClassifier(config.pooler_size, config.clf_hidden_size,
                                                     task_cfg[task_id]["num_labels"])
            elif task_type == "VL-binary-classifier":
                task2clf[task_id] = SimpleClassifier(config.pooler_size * 2, config.clf_hidden_size, 2)
            elif task_type == "VL-tri-classifier":
                task2clf[task_id] = nn.Linear(config.pooler_size, 3)  # for Visual Entailiment tasks
            elif task_type == "VL-logit":
                task2clf[task_id] = nn.Linear(config.pooler_size, 1)
            elif task_type.startswith("V-logit"):
                if task_cfg[task_id].get("num_clf_layers", 1) == 2:
                    task2clf[task_id] = torch.nn.Sequential(
                        nn.Linear(config.v_hidden_size, config.v_hidden_size),
                        GeLU(),
                        torch.nn.Dropout(config.v_attention_probs_dropout_prob, inplace=False),
                        nn.Linear(config.v_hidden_size, 1)
                    )
                else:
                    task2clf[task_id] = nn.Linear(config.v_hidden_size, 1)
            else:
                raise ValueError("Undefined task type: %s" % task_type)

        self.clfs_dict = nn.ModuleDict(task2clf)

        self.fusion_method = config.fusion_method
        self.apply(self.init_weights)

    def forward(
        self,
        input_txt,
        input_imgs,
        image_loc,
        task_id,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        output_all_encoded_layers=False,
        output_all_attention_masks=False,
    ):

        encoded_layers_t, encoded_layers_v, pooled_output_t, pooled_output_v, all_attention_mask = self.bert(
            input_txt,
            input_imgs,
            image_loc,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            output_all_attention_masks=output_all_attention_masks,
        )
        if output_all_encoded_layers:
            sequence_output_t = encoded_layers_t[-1]
            sequence_output_v = encoded_layers_v[-1]
        else:
            sequence_output_t = encoded_layers_t
            sequence_output_v = encoded_layers_v

        linguistic_prediction, vision_prediction = None, None

        if self.fusion_method == "sum":
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == "mul":
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        elif self.fusion_method == "text":
            pooled_output = self.dropout(pooled_output_t)
        elif self.fusion_method == "vl-bert_vqa":
            pooled_output = self.dropout(pooled_output_t)
        elif self.fusion_method == "none":
            pooled_output = None
        else:
            raise ValueError("Invalid fusion method: %s" % self.fusion_method)

        if self.task_cfg[task_id]["type"].startswith("V-logit"):
            vil_prediction = self.clfs_dict[task_id](self.dropout(sequence_output_v)) + (
                (1.0 - image_attention_mask) * -10000.0).unsqueeze(2).to(dtype=next(self.parameters()).dtype)
        elif self.task_cfg[task_id]["type"] == "VL-binary-classifier":
            # NLVR
            vil_prediction = self.clfs_dict[task_id](pooled_output.view(-1, pooled_output.size(1) * 2))
        else:
            vil_prediction = self.clfs_dict[task_id](pooled_output)

        if output_all_encoded_layers:
            return vil_prediction, vision_prediction, linguistic_prediction, all_attention_mask, \
                   encoded_layers_t, encoded_layers_v
        return vil_prediction, vision_prediction, linguistic_prediction, all_attention_mask
