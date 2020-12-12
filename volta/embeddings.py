# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging

import torch
from torch import nn


logger = logging.getLogger(__name__)

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


# ==================================================================================================================== #
#                                                     Text layers                                                      #
# ==================================================================================================================== #
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super(RobertaEmbeddings, self).__init__(config)
        self.padding_idx = 1

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            # Position numbers begin at padding_idx+1. Padding symbols are ignored.
            # cf. fairseq's `utils.make_positions`
            position_ids = torch.arange(
                self.padding_idx + 1,
                seq_length + self.padding_idx + 1,
                dtype=torch.long,
                device=input_ids.device,
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return super(RobertaEmbeddings, self).forward(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids
        )


# ==================================================================================================================== #
#                                                    Vision layers                                                     #
# ==================================================================================================================== #
def coordinate_embeddings(boxes, dim):
    """
    Coordinate embeddings of bounding boxes
    :param boxes: [BS, K, 4] ([x1, y1, x2, y2])
    :param dim: sin/cos embedding dimension
    :return: [BS, K, 4, 2 * dim]
    """

    batch_size, num_boxes, num_loc = boxes.shape

    # transform to (x_c, y_c, w, h) format
    pos = boxes.new_zeros((batch_size, num_boxes, 4))
    pos[:, :, 0] = (boxes[:, :, 0] + boxes[:, :, 2]) / 2 * 100
    pos[:, :, 1] = (boxes[:, :, 1] + boxes[:, :, 3]) / 2 * 100
    pos[:, :, 2] = (boxes[:, :, 2] - boxes[:, :, 0]) * 100
    pos[:, :, 3] = (boxes[:, :, 3] - boxes[:, :, 1]) * 100

    # sin/cos embedding
    dim_mat = 1000 ** (torch.arange(dim, dtype=boxes.dtype, device=boxes.device) / float(dim))
    sin_embedding = (pos.view((batch_size, num_boxes, 4, 1)) / dim_mat.view((1, 1, 1, -1))).sin()
    cos_embedding = (pos.view((batch_size, num_boxes, 4, 1)) / dim_mat.view((1, 1, 1, -1))).cos()

    return torch.cat((sin_embedding, cos_embedding), dim=-1)


class ViLBertImageEmbeddings(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and token_type embeddings.
    """

    def __init__(self, config):
        super(ViLBertImageEmbeddings, self).__init__()

        self.image_embeddings = nn.Linear(config.v_feature_size, config.v_hidden_size)
        self.image_location_embeddings = nn.Linear(config.num_locs, config.v_hidden_size)
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, input_ids, input_loc):
        img_embeddings = self.image_embeddings(input_ids)
        loc_embeddings = self.image_location_embeddings(input_loc)

        embeddings = self.LayerNorm(img_embeddings + loc_embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class LxmertImageEmbeddings(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and token_type embeddings.
    """

    def __init__(self, config):
        super(LxmertImageEmbeddings, self).__init__()

        self.image_embeddings = nn.Linear(config.v_feature_size, config.v_hidden_size)
        self.image_location_embeddings = nn.Linear(config.num_locs, config.v_hidden_size)
        self.ImgLayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.LocLayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, input_ids, input_loc):
        img_embeddings = self.image_embeddings(input_ids)
        loc_embeddings = self.image_location_embeddings(input_loc)

        img_embeddings = self.ImgLayerNorm(img_embeddings)
        loc_embeddings = self.LocLayerNorm(loc_embeddings)

        embeddings = (img_embeddings + loc_embeddings) / 2
        embeddings = self.dropout(embeddings)

        return embeddings


dual_embeddings = {
    "vilbert": ViLBertImageEmbeddings,
    "lxmert": LxmertImageEmbeddings,
}


# ==================================================================================================================== #
#                                                    Bimodal layers                                                    #
# ==================================================================================================================== #
class VLBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(VLBertEmbeddings, self).__init__()

        self.hidden_size = config.hidden_size
        self.with_mvrc_loss = config.visual_target_weights.get("6", 0) > 0
        self.initializer_range = config.initializer_range

        self.v_coordinate_embeddings_dim = config.v_coordinate_embeddings_dim
        self.obj_downsample = torch.nn.Sequential(
            torch.nn.Dropout(config.v_attention_probs_dropout_prob),
            torch.nn.Linear(2 * config.v_feature_size, config.v_hidden_size),
            torch.nn.ReLU(inplace=True),
        )

        self.object_linguistic_embeddings = nn.Embedding(1, config.hidden_size)
        if self.with_mvrc_loss:
            self.object_mask_word_embedding = nn.Embedding(1, config.hidden_size)
        self.object_mask_visual_embedding = nn.Embedding(1, config.v_feature_size)
        self.end_embedding = nn.Embedding(1, config.hidden_size)

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # visual transform
        self.visual_1x1_text = None
        self.visual_1x1_object = None
        if config.v_hidden_size != config.hidden_size:
            self.visual_1x1_text = nn.Linear(config.v_hidden_size, config.hidden_size)
            self.visual_1x1_object = nn.Linear(config.v_hidden_size, config.hidden_size)
        self.visual_ln_text = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.visual_ln_object = BertLayerNorm(config.hidden_size, eps=1e-12)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # init weights
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.obj_downsample[1].weight)
        self.object_mask_visual_embedding.weight.data.fill_(0.0)
        self.object_linguistic_embeddings.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if self.with_mvrc_loss:
            self.object_mask_word_embedding.weight.data.normal_(mean=0.0, std=self.initializer_range)
        self.visual_ln_text.weight.data.fill_(0.0)
        self.visual_ln_object.weight.data.fill_(0.0)
        self.LayerNorm.bias.data.zero_()
        self.LayerNorm.weight.data.fill_(1.0)

    def forward(self, token_ids, image_feat, image_loc, token_type_ids=None, position_ids=None):
        batch_size, num_boxes, _ = image_feat.shape

        mvrc_mask = (image_feat[:, :] == image_feat.new_zeros(image_feat.shape[-1])).sum(-1) == image_feat.shape[-1]
        image_feat[mvrc_mask] = self.object_mask_visual_embedding.weight[0]

        # Geometry Embedding + Appearance Feature
        coord_embed = coordinate_embeddings(image_loc, self.v_coordinate_embeddings_dim)
        feats_to_downsample = torch.cat(
            (coord_embed.view((batch_size * num_boxes, -1)), image_feat.view((batch_size * num_boxes, -1))),
            -1
        )
        final_feats = self.obj_downsample(feats_to_downsample).view(batch_size, num_boxes, -1)  # [BS, K, v_hidden]

        # Token Embedding for vision
        object_visual_embeddings = final_feats
        if self.visual_1x1_object is not None:
            object_visual_embeddings = self.visual_1x1_object(final_feats)
        object_visual_embeddings = self.visual_ln_object(object_visual_embeddings)  # [BS, K, v_hidden]
        object_linguistic_embeddings = self.object_linguistic_embeddings(
            final_feats.new_zeros((batch_size, num_boxes)).long()
        )  # [BS, K, v_hidden]
        if self.with_mvrc_loss:
            object_linguistic_embeddings[mvrc_mask] = self.object_mask_word_embedding.weight[0]
        _zero_id = torch.zeros((batch_size,), dtype=torch.long, device=object_linguistic_embeddings.device)
        object_linguistic_embeddings[:, -1] = self.end_embedding(_zero_id)
        object_vl_embeddings = object_linguistic_embeddings + object_visual_embeddings

        # Token Embedding + Visual Feature Embedding for text
        seq_length = token_ids.size(1)
        text_linguistic_embedding = self.word_embeddings(token_ids)
        text_visual_embeddings = final_feats[:, -1].repeat(1, seq_length).view(batch_size, seq_length, -1)
        if self.visual_1x1_text is not None:
            text_visual_embeddings = self.visual_1x1_text(text_visual_embeddings)
        text_visual_embeddings = self.visual_ln_text(text_visual_embeddings)
        text_vl_embeddings = text_linguistic_embedding + text_visual_embeddings

        # concatenate text and image
        text_mask = token_ids != 0
        text_end = text_mask.sum(1, keepdim=True)
        text_token_type_embeddings = self.token_type_embeddings(token_type_ids)
        object_type_ids = token_type_ids.new_zeros((batch_size, num_boxes)) + 2
        object_token_type_embeddings = self.token_type_embeddings(object_type_ids)

        # position embeddings
        text_position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
        text_position_ids = text_position_ids.unsqueeze(0).expand_as(token_ids)
        text_position_ids[text_position_ids >= text_end] += num_boxes  # FIXME for variable number of objects
        object_position_ids = text_position_ids.new_zeros((batch_size, num_boxes))
        object_position_ids += text_end
        object_position_ids[:, -1] += 1
        text_position_embeddings = self.position_embeddings(text_position_ids)
        object_position_embeddings = self.position_embeddings(object_position_ids)

        embeddings = text_vl_embeddings + text_position_embeddings + text_token_type_embeddings
        v_embeddings = object_vl_embeddings + object_position_embeddings + object_token_type_embeddings
        vl_embeddings = torch.cat((embeddings, v_embeddings), dim=1)
        vl_embeddings = self.LayerNorm(vl_embeddings)
        vl_embeddings = self.dropout(vl_embeddings)
        embeddings, v_embeddings = vl_embeddings.split([embeddings.size(1), v_embeddings.size(1)], dim=1)

        return embeddings, v_embeddings


class VisualBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position, token_type embeddings and visual embeddings.
    """
    def __init__(self, config):
        super(VisualBertEmbeddings, self).__init__()

        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Segment and position embedding for image features
        self.projection = nn.Linear(config.v_feature_size, config.hidden_size)
        self.token_type_embeddings_visual = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings_visual = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.special_initialize()

    def special_initialize(self):
        # This is a bit unorthodox. The better way might be to add an initializer to AllenNLP.
        # This function is used to init the token_type_embeddings_visual and position_embedding_visual, just in case.
        self.token_type_embeddings_visual.weight = torch.nn.Parameter(
            copy.deepcopy(self.token_type_embeddings.weight.data), requires_grad=True)
        self.position_embeddings_visual.weight = torch.nn.Parameter(
            copy.deepcopy(self.position_embeddings.weight.data), requires_grad=True)

    def forward(self, token_ids, image_feat, image_loc, token_type_ids=None, position_ids=None):
        batch_size, num_boxes, _ = image_feat.shape
        seq_length = token_ids.size(1)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(token_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(token_ids)

        words_embeddings = self.word_embeddings(token_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        if image_feat is not None:
            visual_embeddings = self.projection(image_feat)
            visual_embeddings_type = token_type_ids.new_zeros((batch_size, num_boxes)) + 1
            token_type_embeddings_visual = self.token_type_embeddings_visual(visual_embeddings_type)

            image_text_alignment = None  # FIXME for VCR
            if image_text_alignment is not None:
                # image_text_alignment = Batch x image_length x alignment_number.
                # Each element denotes the position of the word corresponding to the image feature. -1 is padding value.
                image_text_alignment_mask = (image_text_alignment != -1).long()
                # Get rid of the -1.
                image_text_alignment = image_text_alignment_mask * image_text_alignment

                # position_embeddings_visual = Batch x image_length x alignment length x dim
                position_embeddings_visual = self.position_embeddings(image_text_alignment) * \
                    image_text_alignment_mask.to(dtype=next(self.parameters()).dtype).unsqueeze(-1)
                position_embeddings_visual = position_embeddings_visual.sum(2)

                # We want to average along the alignment_number dimension.
                image_text_alignment_mask = image_text_alignment_mask.to(dtype=next(self.parameters()).dtype).sum(2)
                image_text_alignment_mask[image_text_alignment_mask == 0] = 1  # Avoid divide by zero error
                position_embeddings_visual = position_embeddings_visual / image_text_alignment_mask.unsqueeze(-1)

                position_ids_visual = torch.zeros(*visual_embeddings.size()[:-1], dtype=torch.long).cuda()

                # When fine-tuning the detector, the image_text_alignment is sometimes padded too long.
                if position_embeddings_visual.size(1) != visual_embeddings.size(1):
                    assert(position_embeddings_visual.size(1) >= visual_embeddings.size(1))
                    position_embeddings_visual = position_embeddings_visual[:, :visual_embeddings.size(1), :]

                position_embeddings_visual = position_embeddings_visual + \
                    self.position_embeddings_visual(position_ids_visual)
            else:
                position_ids_visual = torch.zeros(*visual_embeddings.size()[:-1], dtype=torch.long).cuda()
                position_embeddings_visual = self.position_embeddings_visual(position_ids_visual)

            v_embeddings = visual_embeddings + position_embeddings_visual + token_type_embeddings_visual

            # Concat the two:
            vl_embeddings = torch.cat((embeddings, v_embeddings), dim=1)  # concat visual embeddings after attentions
            vl_embeddings = self.LayerNorm(vl_embeddings)
            vl_embeddings = self.dropout(vl_embeddings)
            embeddings, v_embeddings = vl_embeddings.split([embeddings.size(1), v_embeddings.size(1)], dim=1)
        else:
            v_embeddings = None
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)

        return embeddings, v_embeddings


class UniterEmbeddings(nn.Module):
    """Construct the embeddings from word, position, token_type embeddings and visual embeddings.
    """
    def __init__(self, config):
        super(UniterEmbeddings, self).__init__()

        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.image_embeddings = nn.Linear(config.v_feature_size, config.v_hidden_size)
        self.image_location_embeddings = nn.Linear(config.num_locs, config.v_hidden_size)
        self.image_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.image_location_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.v_LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.v_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.special_initialize()

    def special_initialize(self):
        # This function is used to init v_LayerNorm as LayerNorm
        self.v_LayerNorm.weight = torch.nn.Parameter(copy.deepcopy(self.LayerNorm.weight.data), requires_grad=True)
        self.v_LayerNorm.bias = torch.nn.Parameter(copy.deepcopy(self.LayerNorm.bias.data), requires_grad=True)

    def forward(self, token_ids, image_feat, image_loc, token_type_ids=None, position_ids=None):
        batch_size, num_boxes, _ = image_feat.shape
        seq_length = token_ids.size(1)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(token_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(token_ids)

        words_embeddings = self.word_embeddings(token_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        img_embeddings = self.image_layer_norm(self.image_embeddings(image_feat))
        loc_embeddings = self.image_location_layer_norm(self.image_location_embeddings(image_loc))
        img_type_ids = torch.ones_like(image_feat[:, :, 0].long())
        v_token_type_embeddings = self.token_type_embeddings(img_type_ids)
        v_embeddings = img_embeddings + loc_embeddings + v_token_type_embeddings
        v_embeddings = self.v_LayerNorm(v_embeddings)
        v_embeddings = self.v_dropout(v_embeddings)

        return embeddings, v_embeddings


shared_embeddings = {
    "vl-bert": VLBertEmbeddings,
    "visualbert": VisualBertEmbeddings,
    "uniter": UniterEmbeddings,
}
