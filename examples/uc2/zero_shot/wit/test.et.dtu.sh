#!/bin/bash

TASK=20
lang=et
MODEL=uc2
MODEL_CONFIG=uc2_base
TASKS_CONFIG=iglue_test_tasks_boxes36
TRTASK=RetrievalWIT
TETASK=RetrievalWIT${lang}
TEXT_PATH=data/wit/annotations/test_${lang}.jsonl
FEAT_PATH=data/wit/features/wit_test_boxes36.lmdb
PRETRAINED=checkpoints/iglue/zero_shot/wit/${MODEL}/${TRTASK}_${MODEL_CONFIG}/pytorch_model_best.bin
OUTPUT_DIR=results/iglue/zero_shot/wit/${MODEL}/${TRTASK}_${MODEL_CONFIG}/$TETASK/test

source envs/iglue/bin/activate

cd ../../../..
python eval_retrieval.py \
  --bert_model huggingface/xlm-roberta-base --config_file config/${MODEL_CONFIG}.json \
  --from_pretrained ${PRETRAINED} \
  --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --caps_per_image 1 --split test_${lang} --batch_size 1 \
  --output_dir ${OUTPUT_DIR} --val_annotations_jsonpath ${TEXT_PATH} --val_features_lmdbpath ${FEAT_PATH}

deactivate
