#!/bin/bash

TASK=8
SHOT=25
LANG=es
MODEL=uc2
MODEL_CONFIG=uc2_base
TASKS_CONFIG=iglue_test_tasks_boxes36
TRTASK=RetrievalxFlickrCO${LANG}_${SHOT}
TETASK=RetrievalxFlickrCO${LANG}
TEXT_PATH=data/xFlickrCO/annotations/${LANG}/test.jsonl
FEAT_PATH=data/xFlickrCO/features/xflickrco-test_boxes36.lmdb

here=$(pwd)

source envs/iglue/bin/activate

cd ../../../../../..

best=-1
best_lr=-1
for lr in 1e-4 5e-5 1e-5; do
  f=${here}/train.${lr}.log
  s=`tail -n1 $f | cut -d ' ' -f 4`
  d=$(echo "$s>$best" | bc)
  if [[ $d -eq 1 ]]; then
    best=$s
    best_lr=$lr
  fi
done
echo "Best lr: " $best_lr
PRETRAINED=checkpoints/iglue/few_shot/xflickrco/${TRTASK}/${MODEL}/${best_lr}/RetrievalFlickr30k_${MODEL_CONFIG}/pytorch_model_best.bin
OUTPUT_DIR=results/iglue/few_shot/xflickrco/${MODEL}/${best_lr}/${TRTASK}_${MODEL_CONFIG}/$TETASK/test

python eval_retrieval.py \
  --bert_model huggingface/xlm-roberta-base --config_file config/${MODEL_CONFIG}.json \
  --from_pretrained ${PRETRAINED} \
  --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
  --split test_${LANG} --batch_size 1 \
  --caps_per_image 1 --val_annotations_jsonpath ${TEXT_PATH} --val_features_lmdbpath ${FEAT_PATH} \
  --output_dir ${OUTPUT_DIR} \

deactivate
