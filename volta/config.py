# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import copy
from io import open


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(
        self,
        vocab_size_or_config_json_file,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        pooler_size=768,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        num_locs=5,
        v_coordinate_embeddings_dim=None,
        add_global_imgfeat=None,
        image_embeddings="vilbert",
        initializer_range=0.02,
        v_feature_size=2048,
        v_hidden_size=768,
        v_num_attention_heads=12,
        v_intermediate_size=3072,
        v_pooler_size=1024,
        v_attention_probs_dropout_prob=0.1,
        v_hidden_act="gelu",
        v_hidden_dropout_prob=0.1,
        v_initializer_range=0.2,
        visual_target_weights={"0": 1},
        fixed_layers=[],
        fusion_method="mul",
        objective=0,
        clf_hidden_size=1536,
        image_head_ln=True,
        model="bert",
        visualization=False,
        tt_attn_sublayers=[],
        tv_attn_sublayers=[],
        vt_attn_sublayers=[],
        vv_attn_sublayers=[],
        t_ff_sublayers=[],
        v_ff_sublayers=[],
        shared_sublayers=[],
        single_ln_sublayers=[],
        sublayer2attn_hidden_size={},
        sublayer2num_attention_heads={},
        sublayer2intermediate_size={},
        sublayer2v_attn_hidden_size={},
        sublayer2v_num_attention_heads={},
        sublayer2v_intermediate_size={},
        bert_layer2attn_sublayer={},
        bert_layer2ff_sublayer={},
    ):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding="utf-8") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            # Text
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.pooler_size = pooler_size
            # Vision
            self.num_locs = num_locs
            self.v_coordinate_embeddings_dim = v_coordinate_embeddings_dim
            self.add_global_imgfeat = add_global_imgfeat
            self.image_embeddings = image_embeddings
            self.v_feature_size = v_feature_size
            self.v_hidden_size = v_hidden_size
            self.v_num_attention_heads = v_num_attention_heads
            self.v_intermediate_size = v_intermediate_size
            self.v_attention_probs_dropout_prob = v_attention_probs_dropout_prob
            self.v_hidden_act = v_hidden_act
            self.v_hidden_dropout_prob = v_hidden_dropout_prob
            self.v_initializer_range = v_initializer_range
            self.v_pooler_size = v_pooler_size
            # Text-Vision
            self.tt_attn_sublayers = tt_attn_sublayers
            self.tv_attn_sublayers = tv_attn_sublayers
            self.vt_attn_sublayers = vt_attn_sublayers
            self.vv_attn_sublayers = vv_attn_sublayers
            self.t_ff_sublayers = t_ff_sublayers
            self.v_ff_sublayers = v_ff_sublayers
            self.shared_sublayers = shared_sublayers
            self.single_ln_sublayers = single_ln_sublayers
            self.sublayer2attn_hidden_size = sublayer2attn_hidden_size
            self.sublayer2num_attention_heads = sublayer2num_attention_heads
            self.sublayer2intermediate_size = sublayer2intermediate_size
            self.sublayer2v_attn_hidden_size = sublayer2v_attn_hidden_size
            self.sublayer2v_num_attention_heads = sublayer2v_num_attention_heads
            self.sublayer2v_intermediate_size = sublayer2v_intermediate_size
            self.bert_layer2attn_sublayer = bert_layer2attn_sublayer
            self.bert_layer2ff_sublayer = bert_layer2ff_sublayer
            self.image_head_ln = image_head_ln
            # Else
            self.visual_target_weights = visual_target_weights
            self.fixed_layers = fixed_layers
            self.model = model
            # Pre-training
            self.fusion_method = fusion_method
            self.objective = objective
            # Fine-tuning
            self.clf_hidden_size = clf_hidden_size
            self.visualization = visualization
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)"
            )

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
