#!/bin/bash

TASK=18
MODEL=ctrl_uniter
MODEL_CONFIG=ctrl_uniter_base
TASKS_CONFIG=cross-modal-ablation_test_tasks
PRETRAINED=checkpoints/${MODEL}/${MODEL_CONFIG}/pytorch_model_9.bin
OUTPUT_DIR=results/cross-modal-ablation/flickr30kentities_lang4vis/${MODEL}

source activate volta

cd ../../../../..
python ablate_lang4vis.py \
        --config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
        --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --split val \
        --output_dir ${OUTPUT_DIR} --dump_results --masking none

conda deactivate
