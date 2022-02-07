# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import copy
from io import open


m3p_id2lang = {
     0: 'af',
     1: 'als',
     2: 'am',
     3: 'an',
     4: 'ang',
     5: 'ar',
     6: 'arz',
     7: 'ast',
     8: 'az',
     9: 'bar',
     10: 'be',
     11: 'bg',
     12: 'bn',
     13: 'br',
     14: 'bs',
     15: 'ca',
     16: 'ceb',
     17: 'ckb',
     18: 'cs',
     19: 'cy',
     20: 'da',
     21: 'de',
     22: 'el',
     23: 'en',
     24: 'eo',
     25: 'es',
     26: 'et',
     27: 'eu',
     28: 'fa',
     29: 'fi',
     30: 'fr',
     31: 'fy',
     32: 'ga',
     33: 'gan',
     34: 'gl',
     35: 'gu',
     36: 'he',
     37: 'hi',
     38: 'hr',
     39: 'hu',
     40: 'hy',
     41: 'ia',
     42: 'id',
     43: 'is',
     44: 'it',
     45: 'ja',
     46: 'jv',
     47: 'ka',
     48: 'kk',
     49: 'kn',
     50: 'ko',
     51: 'ku',
     52: 'la',
     53: 'lb',
     54: 'lt',
     55: 'lv',
     56: 'mk',
     57: 'ml',
     58: 'mn',
     59: 'mr',
     60: 'ms',
     61: 'my',
     62: 'nds',
     63: 'ne',
     64: 'nl',
     65: 'nn',
     66: 'no',
     67: 'oc',
     68: 'pl',
     69: 'pt',
     70: 'ro',
     71: 'ru',
     72: 'scn',
     73: 'sco',
     74: 'sh',
     75: 'si',
     76: 'simple',
     77: 'sk',
     78: 'sl',
     79: 'sq',
     80: 'sr',
     81: 'sv',
     82: 'sw',
     83: 'ta',
     84: 'te',
     85: 'th',
     86: 'tl',
     87: 'tr',
     88: 'tt',
     89: 'uk',
     90: 'ur',
     91: 'uz',
     92: 'vi',
     93: 'war',
     94: 'wuu',
     95: 'yi',
     96: 'zh',
     97: 'zh_classical',
     98: 'zh_min_nan',
     99: 'zh_yue',
}

m3p_lang2id = {
     'af': 0,
     'als': 1,
     'am': 2,
     'an': 3,
     'ang': 4,
     'ar': 5,
     'arz': 6,
     'ast': 7,
     'az': 8,
     'bar': 9,
     'be': 10,
     'bg': 11,
     'bn': 12,
     'br': 13,
     'bs': 14,
     'ca': 15,
     'ceb': 16,
     'ckb': 17,
     'cs': 18,
     'cy': 19,
     'da': 20,
     'de': 21,
     'el': 22,
     'en': 23,
     'eo': 24,
     'es': 25,
     'et': 26,
     'eu': 27,
     'fa': 28,
     'fi': 29,
     'fr': 30,
     'fy': 31,
     'ga': 32,
     'gan': 33,
     'gl': 34,
     'gu': 35,
     'he': 36,
     'hi': 37,
     'hr': 38,
     'hu': 39,
     'hy': 40,
     'ia': 41,
     'id': 42,
     'is': 43,
     'it': 44,
     'ja': 45,
     'jv': 46,
     'ka': 47,
     'kk': 48,
     'kn': 49,
     'ko': 50,
     'ku': 51,
     'la': 52,
     'lb': 53,
     'lt': 54,
     'lv': 55,
     'mk': 56,
     'ml': 57,
     'mn': 58,
     'mr': 59,
     'ms': 60,
     'my': 61,
     'nds': 62,
     'ne': 63,
     'nl': 64,
     'nn': 65,
     'no': 66,
     'oc': 67,
     'pl': 68,
     'pt': 69,
     'ro': 70,
     'ru': 71,
     'scn': 72,
     'sco': 73,
     'sh': 74,
     'si': 75,
     'simple': 76,
     'sk': 77,
     'sl': 78,
     'sq': 79,
     'sr': 80,
     'sv': 81,
     'sw': 82,
     'ta': 83,
     'te': 84,
     'th': 85,
     'tl': 86,
     'tr': 87,
     'tt': 88,
     'uk': 89,
     'ur': 90,
     'uz': 91,
     'vi': 92,
     'war': 93,
     'wuu': 94,
     'yi': 95,
     'zh': 96,
     'zh_classical': 97,
     'zh_min_nan': 98,
     'zh_yue': 99,
}


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
        do_lower_case=True,
        pad_token_id=0,
        layer_norm_eps=1e-12,
        num_locs=5,
        max_boxes=36,
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
        qa_task_weight=0.0,
        qa_num_answers=0,
        fixed_layers=[],
        fusion_method="mul",
        fusion_act="relu",
        objective=0,
        clf_hidden_size=1536,
        m_encoder=None,  # FIXME?
        m_layer=0,  # FIXME?
        v_layers=[],
        has_mapping="linear",
        has_mapping_bias=False,
        load_x_model=False,
        fixed_embs=None,
        image_head_ln=True,
        bert_model="bert-base-uncased",
        itm_dim=2,
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
        norm_embeddings=False,
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
            self.pad_token_id = pad_token_id
            self.layer_norm_eps = layer_norm_eps
            self.do_lower_case = do_lower_case
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
            self.qa_task_weight = qa_task_weight
            self.qa_num_answers = qa_num_answers
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
            self.itm_dim = itm_dim
            # Else
            self.visual_target_weights = visual_target_weights
            self.fixed_layers = fixed_layers
            self.bert_model = bert_model
            self.m_encoder = m_encoder  # FIXME?
            self.m_layer = m_layer  # FIXME?
            self.v_layers = v_layers
            self.has_mapping = has_mapping
            self.has_mapping_bias = has_mapping_bias
            self.load_x_model = load_x_model
            self.fixed_embs = fixed_embs
            self.norm_embeddings = norm_embeddings
            # Pre-training
            self.fusion_method = fusion_method
            self.fusion_act = fusion_act
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


class M3PConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(
        self,
        vocab_size_or_config_json_file,
        hidden_size=768,
        n_heads=12,
        intermediate_size=3072,
        pooler_size=768,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        pad_token_id=0,
        layer_norm_eps=1e-12,
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
        norm_embeddings=False,
        visual_target_weights={"0": 1},
        fixed_layers=[],
        fusion_act="relu",
        objective=0,
        clf_hidden_size=1536,
        m_encoder=None,
        m_layer=0,
        v_layers=[],
        has_mapping="linear",
        has_mapping_bias=False,
        load_x_model=False,
        fixed_embs=None,
        image_head_ln=True,
        model="bert",
        itm_dim=2,
        visualization=False,
        n_langs=100,
        n_words=250002,
        eos_index=2,
        pad_index=1,
        id2lang=m3p_id2lang,
        lang2id=m3p_lang2id,
        emb_dim=768,
        n_layers=12,
        dropout=0.1,
        attention_dropout=0.1,
        sinusoidal_embeddings=False,
        refine_layers=6,
        attention_setting='v1',
        use_externel_att=False,
        gelu_activation=True,
        max_boxes=100,
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
            self.n_heads = n_heads
            self.n_layers = n_layers
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.pad_token_id = pad_token_id
            self.layer_norm_eps = layer_norm_eps
            self.initializer_range = initializer_range
            self.pooler_size = pooler_size
            self.n_langs = n_langs
            self.n_words = n_words
            self.eos_index = eos_index
            self.pad_index = pad_index
            self.id2lang = id2lang
            self.lang2id = lang2id
            self.dropout = dropout
            self.attention_dropout = attention_dropout
            self.sinusoidal_embeddings = sinusoidal_embeddings
            self.refine_layers = refine_layers
            self.attention_setting = attention_setting
            self.use_externel_att = use_externel_att
            self.gelu_activation = gelu_activation
            # Vision
            self.num_locs = num_locs
            self.max_boxes = max_boxes
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
            self.emb_dim = emb_dim
            self.norm_embeddings = norm_embeddings
            self.max_boxes = max_boxes
            self.image_head_ln = image_head_ln
            self.itm_dim = itm_dim
            # Else
            self.visual_target_weights = visual_target_weights
            self.fixed_layers = fixed_layers
            self.model = model
            self.m_encoder = m_encoder  # FIXME?
            self.m_layer = m_layer  # FIXME?
            self.v_layers = v_layers
            self.has_mapping = has_mapping
            self.has_mapping_bias = has_mapping_bias
            self.load_x_model = load_x_model
            self.fixed_embs = fixed_embs
            # Pre-training
            self.fusion_act = fusion_act
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
        config = M3PConfig(vocab_size_or_config_json_file=-1)
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
