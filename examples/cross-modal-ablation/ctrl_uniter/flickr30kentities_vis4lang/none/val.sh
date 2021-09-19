#!/bin/bash

TASK=19
MODEL=ctrl_uniter
MODEL_CONFIG=ctrl_uniter_base
TASKS_CONFIG=cross-modal-ablation_test_tasks
PRETRAINED=checkpoints/${MODEL}/${MODEL_CONFIG}/pytorch_model_9.bin
OUTPUT_DIR=results/cross-modal-ablation/flickr30kentities_vis4lang/${MODEL}

source activate volta

cd ../../../../..
python ablate_vis4lang.py \
        --bert_model bert-base-uncased --config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
        --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --split val \
        --output_dir ${OUTPUT_DIR} --dump_results --masking none

conda deactivate
