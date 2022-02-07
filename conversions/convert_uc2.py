import os
import sys
import argparse
sys.path.append("../")

import torch
from volta.config import BertConfig
from volta.encoders import BertForVLPreTraining

XLMR_path = "/science/image/nlp-datasets/emanuele/huggingface/xlm-roberta-base"


# Inputs
parser = argparse.ArgumentParser()
parser.add_argument("--input_fn", type=str, default="UC2_DATA/pretrain/uc2/ckpt/model_step_200000.pt")
parser.add_argument("--output_fn", type=str, default="uc2_checkpoint_200000.bin")
parser.add_argument("--verbose", action="store_true", default=False)
args = parser.parse_args()

# Load original checkpoint
original_ckpt = torch.load(args.input_fn, map_location="cpu")

# Create corresponding VOLTA model
config_file = "../config/uc2_base.json"
config = BertConfig.from_json_file(config_file)
model = BertForVLPreTraining.from_pretrained(XLMR_path, config=config, default_gpu=True, from_hf=True)
trg_dict = model.state_dict()

# Map original parameters onto VOLTA ones
trg_keys = set(trg_dict.keys())
volta2original = dict()
for k in original_ckpt.keys():
    ln = str(k)
    ln = ln.replace("roberta", "bert")
    
    ln = ln.replace("img_embeddings", "embeddings")
    ln = ln.replace("img_linear", "image_embeddings")
    ln = ln.replace("pos_linear", "image_location_embeddings")
    ln = ln.replace("img_layer_norm", "image_layer_norm")
    ln = ln.replace("pos_layer_norm", "image_location_layer_norm")
    
    ln = ln.replace('attention.self', 'attention_self')
    ln = ln.replace('attention.output', 'attention_output')
    if '.layer.' in ln:
        num = int(ln.split(".")[3])
        new = 2*num + ('.intermediate.' in ln or '.output.' in ln)
        ln = ln.replace(f".{num}.", f".{new}.")
        
    ln = ln.replace("pooler", "t_pooler")
    ln = ln.replace("cls.dense", "cls.predictions.transform.dense")
    ln = ln.replace("cls.layer_norm", "cls.predictions.transform.LayerNorm")
    ln = ln.replace("cls.bias", "cls.predictions.bias")
    ln = ln.replace("cls.decoder", "cls.predictions.decoder")
    ln = ln.replace("itm_output", "cls.bi_seq_relationship")
    
    if ln not in trg_keys:
        if args.verbose:
            print("[OMITTED]", k)
    else:
        volta2original[ln] = k
    
# Apply mapping
for trg, src in volta2original.items():
    if args.verbose:
        print(trg, '<-', src)
    assert trg_dict[trg].shape == original_ckpt[src].shape
    trg_dict[trg] = original_ckpt[src]
model.load_state_dict(trg_dict)

# Save checkpoint of VOLTA model
torch.save(model.state_dict(), args.output_fn)


# OMITTED:
# [OMITTED] roberta.img_embeddings.mask_embedding.weight
# [OMITTED] cls.decoder.bias
# [OMITTED] vis_cls.bias
# [OMITTED] vis_cls.dense.weight
# [OMITTED] vis_cls.dense.bias
# [OMITTED] vis_cls.layer_norm.weight
# [OMITTED] vis_cls.layer_norm.bias
# [OMITTED] vis_cls.decoder.weight
# [OMITTED] vis_cls.decoder.bias
# [OMITTED] feat_regress.weight
# [OMITTED] feat_regress.bias
# [OMITTED] feat_regress.net.0.weight
# [OMITTED] feat_regress.net.0.bias
# [OMITTED] feat_regress.net.2.weight
# [OMITTED] feat_regress.net.2.bias
# [OMITTED] region_classifier.net.0.weight
# [OMITTED] region_classifier.net.0.bias
# [OMITTED] region_classifier.net.2.weight
# [OMITTED] region_classifier.net.2.bias
# [OMITTED] region_classifier.net.3.weight
# [OMITTED] region_classifier.net.3.bias
# 
# roberta.img_embeddings.mask_embedding.weight 	 bert.embeddings.mask_embedding.weight
# cls.decoder.bias 								 cls.predictions.decoder.bias
# vis_cls.bias 									 vis_cls.predictions.bias
# vis_cls.dense.weight 							 vis_cls.predictions.transform.dense.weight
# vis_cls.dense.bias 							 vis_cls.predictions.transform.dense.bias
# vis_cls.layer_norm.weight 					 vis_cls.predictions.transform.LayerNorm.weight
# vis_cls.layer_norm.bias 						 vis_cls.predictions.transform.LayerNorm.bias
# vis_cls.decoder.weight 						 vis_cls.predictions.decoder.weight
# vis_cls.decoder.bias 							 vis_cls.predictions.decoder.bias
# feat_regress.weight 							 feat_regress.weight
# feat_regress.bias 							 feat_regress.bias
# feat_regress.net.0.weight 					 feat_regress.net.0.weight
# feat_regress.net.0.bias 						 feat_regress.net.0.bias
# feat_regress.net.2.weight 					 feat_regress.net.2.weight
# feat_regress.net.2.bias 						 feat_regress.net.2.bias
# region_classifier.net.0.weight 				 region_classifier.net.0.weight
# region_classifier.net.0.bias 					 region_classifier.net.0.bias
# region_classifier.net.2.weight 				 region_classifier.net.2.weight
# region_classifier.net.2.bias 					 region_classifier.net.2.bias
# region_classifier.net.3.weight 				 region_classifier.net.3.weight
# region_classifier.net.3.bias 					 region_classifier.net.3.bias