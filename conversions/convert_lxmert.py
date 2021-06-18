import os
import sys
import argparse
sys.path.append("../")

import torch
from volta.config import BertConfig
from volta.encoders import BertForVLPreTraining


# Inputs
parser = argparse.ArgumentParser()
parser.add_argument("--input_fn", type=str, default="Epoch20_LXRT.pth")
parser.add_argument("--output_fn", type=str, default="lxmert_checkpoint_19.bin")
parser.add_argument("--verbose", action="store_true", default=False)
args = parser.parse_args()

# Load original checkpoint
original_ckpt = torch.load(args.input_fn, map_location="cpu")

# Create corresponding VOLTA model
config_file = "../config/original_lxmert.json"
config = BertConfig.from_json_file(config_file)
model = BertForVLPreTraining.from_pretrained("bert-base-uncased", config=config, default_gpu=True, from_hf=True)
trg_dict = model.state_dict()

# Map original parameters onto VOLTA ones
first_xlayer = config.tv_attn_sublayers[0]
volta2original = dict()
for k in original_ckpt.keys():
    ln = k.replace('module.', '')
    ln = ln.replace("encoder.visn_fc", "v_embeddings")
    ln = ln.replace("visn_fc", "image_embeddings")
    ln = ln.replace("visn_layer_norm", "ImgLayerNorm")
    ln = ln.replace("box_fc", "image_location_embeddings")
    ln = ln.replace("box_layer_norm", "LocLayerNorm")
    
    ln = ln.replace('attention.self', 'attention_self')
    ln = ln.replace('attention.output', 'attention_output')
    if '.layer.' in ln:
        num = int(ln.split(".")[3])
        new = 2*num + ('.intermediate.' in ln or '.output.' in ln)
        ln = ln.replace(f".{num}.", f".{new}.")
    elif "r_layers" in ln:
        num = int(ln.split(".")[3])
        new = 2*num + ('.intermediate.' in ln or '.output.' in ln)
        ln = ln.replace(f"r_layers.{num}.", f"layer.{new}.")
        ln = ln.replace('.query.', '.v_query.')
        ln = ln.replace('.key.', '.v_key.')
        ln = ln.replace('.value.', '.v_value.')
        ln = ln.replace("dense", "v_dense")
        ln = ln.replace('.LayerNorm.', '.v_LayerNorm.')
    elif "x_layers" in ln:
        num = int(ln.split(".")[3])
        new = 3*num + first_xlayer
        if '.visual_attention.' in ln:
            ln = ln.replace(f"x_layers.{num}.visual_attention.att", f"layer.{new}.attention_self")
            lnv = ln.replace('.query.', '.v_query.')
            lnv = lnv.replace('.key.', '.v_key.')
            lnv = lnv.replace('.value.', '.v_value.')
            volta2original[lnv] = k
        elif '.visual_attention_output.' in ln:
            ln = ln.replace(f"x_layers.{num}.visual_attention_output", f"layer.{new}.attention_output")
            lnv = ln.replace('.dense.', '.v_dense.')
            lnv = lnv.replace('.LayerNorm.', '.v_LayerNorm.')
            volta2original[lnv] = k
        elif '.lang_self_att.' in ln:
            new += 1
            ln = ln.replace(f"x_layers.{num}.lang_self_att.self", f"layer.{new}.attention_self")
            ln = ln.replace(f"x_layers.{num}.lang_self_att.output", f"layer.{new}.attention_output")
        elif '.visn_self_att' in ln:
            new += 1
            ln = ln.replace(f"x_layers.{num}.visn_self_att.self", f"layer.{new}.attention_self")
            ln = ln.replace(f"x_layers.{num}.visn_self_att.output", f"layer.{new}.attention_output")
            ln = ln.replace('.query.', '.v_query.')
            ln = ln.replace('.key.', '.v_key.')
            ln = ln.replace('.value.', '.v_value.')
            ln = ln.replace('.dense.', '.v_dense.')
            ln = ln.replace('.LayerNorm.', '.v_LayerNorm.')
        elif '.lang_inter.' in ln:
            new += 2
            ln = ln.replace(f"x_layers.{num}.lang_inter.", f"layer.{new}.intermediate.")
        elif '.visn_inter.' in ln:
            new += 2
            ln = ln.replace(f"x_layers.{num}.visn_inter.", f"layer.{new}.intermediate.")
            ln = ln.replace('.dense.', '.v_dense.')
        elif '.lang_output.' in ln:
            new += 2
            ln = ln.replace(f"x_layers.{num}.lang_output.", f"layer.{new}.output.")
        elif '.visn_output.' in ln:
            new += 2
            ln = ln.replace(f"x_layers.{num}.visn_output.", f"layer.{new}.output.")
            ln = ln.replace('.LayerNorm.', '.v_LayerNorm.')
            ln = ln.replace('.dense.', '.v_dense.')
    
    ln = ln.replace("seq_relationship", "bi_seq_relationship")
    ln = ln.replace("pooler", "t_pooler")
    ln = ln.replace("answer_head", "cls.qaPredictions")
    ln = ln.replace("obj_predict_head", "cls.imagePredictions")
    ln = ln.replace("decoder_dict.obj", "decoder_dict.3")
    ln = ln.replace("decoder_dict.attr", "decoder_dict.4")
    ln = ln.replace("decoder_dict.feat", "decoder_dict.5")

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
