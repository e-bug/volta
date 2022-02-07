#!/bin/bash

TASK=15
LANG=zh
MODEL=uc2
MODEL_CONFIG=uc2_base
TASKS_CONFIG=iglue_test_tasks_boxes36
TRTASK=GQA
TETASK=xGQA${LANG}
TEXT_PATH=data/xGQA/annotations/few_shot/${LANG}/test.json
PRETRAINED=checkpoints/iglue/zero_shot/xgqa/${MODEL}/${TRTASK}_${MODEL_CONFIG}/pytorch_model_best.bin
OUTPUT_DIR=results/iglue/zero_shot/xgqa/${MODEL}/${TRTASK}_${MODEL_CONFIG}/$TETASK/test

source envs/iglue/bin/activate

cd ../../../..
python eval_task.py \
  --bert_model huggingface/xlm-roberta-base --config_file config/${MODEL_CONFIG}.json \
  --from_pretrained ${PRETRAINED} \
  --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
  --split test_${LANG} \
  --output_dir ${OUTPUT_DIR} --val_annotations_jsonpath ${TEXT_PATH}
python scripts/GQA_score.py \
  --preds_file ${OUTPUT_DIR}/pytorch_model_best.bin-/test_${LANG}_result.json \
  --truth_file $TEXT_PATH

deactivate
