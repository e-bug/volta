#!/bin/bash

TASK=12
MODEL=uc2
MODEL_CONFIG=uc2_base
TRTASK=NLVR2
TETASK=MaRVLzh
TASKS_CONFIG=iglue_test_tasks_boxes36
TEXT_PATH=data/marvl/zero_shot/annotations/marvl-zh.jsonl
FEAT_PATH=data/marvl/zero_shot/features/marvl-zh_boxes36.lmdb
PRETRAINED=checkpoints/iglue/zero_shot/marvl/${MODEL}/${TRTASK}_${MODEL_CONFIG}/pytorch_model_best.bin
OUTPUT_DIR=results/iglue/zero_shot/marvl/${MODEL}/${TRTASK}_${MODEL_CONFIG}/$TETASK/test

source envs/iglue/bin/activate

cd ../../../..
python eval_task.py \
  --config_file config/${MODEL_CONFIG}.json \
  --from_pretrained ${PRETRAINED} \
  --val_annotations_jsonpath ${TEXT_PATH} --val_features_lmdbpath ${FEAT_PATH} \
  --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --split test \
  --output_dir ${OUTPUT_DIR}

deactivate
