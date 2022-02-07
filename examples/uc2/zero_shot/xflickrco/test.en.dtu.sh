#!/bin/bash

TASK=8
LANG=en
MODEL=uc2
MODEL_CONFIG=uc2_base
TASKS_CONFIG=iglue_test_tasks_boxes36
TRTASK=RetrievalFlickr30k
TETASK=RetrievalxFlickrCO${LANG}
TEXT_PATH=data/xFlickrCO/annotations/${LANG}/test.jsonl
FEAT_PATH=data/xFlickrCO/features/xflickrco-test_boxes36.lmdb
PRETRAINED=checkpoints/iglue/zero_shot/xflickrco/${MODEL}/${TRTASK}_${MODEL_CONFIG}/pytorch_model_best.bin
OUTPUT_DIR=results/iglue/zero_shot/xflickrco/${MODEL}/${TRTASK}_${MODEL_CONFIG}/$TETASK/test

source envs/iglue/bin/activate

cd ../../../..
python eval_retrieval.py \
  --bert_model huggingface/xlm-roberta-base --config_file config/${MODEL_CONFIG}.json \
  --from_pretrained ${PRETRAINED} --num_val_workers 0 \
  --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --split test_${LANG} --batch_size 1 \
  --caps_per_image 1 --val_annotations_jsonpath ${TEXT_PATH} --val_features_lmdbpath ${FEAT_PATH} \
  --output_dir ${OUTPUT_DIR}
deactivate
