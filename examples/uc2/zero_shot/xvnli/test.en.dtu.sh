#!/bin/bash

TASK=19
MODEL=uc2
MODEL_CONFIG=uc2_base
TASKS_CONFIG=iglue_test_tasks_boxes36
TRTASK=XVNLI
TETASK=XVNLI
PRETRAINED=checkpoints/iglue/zero_shot/xvnli/${MODEL}/${TRTASK}_${MODEL_CONFIG}/pytorch_model_best.bin
OUTPUT_DIR=results/iglue/zero_shot/xvnli/${MODEL}/${TRTASK}_${MODEL_CONFIG}/$TETASK/test

source envs/iglue/bin/activate

cd ../../../..
python eval_task.py \
  --bert_model huggingface/xlm-roberta-base --config_file config/${MODEL_CONFIG}.json \
  --from_pretrained ${PRETRAINED} \
  --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --split test \
  --output_dir ${OUTPUT_DIR}

deactivate
