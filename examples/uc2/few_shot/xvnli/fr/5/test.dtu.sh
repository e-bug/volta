#!/bin/bash

TASK=19
SHOT=5
LANG=fr
MODEL=uc2
MODEL_CONFIG=uc2_base
TASKS_CONFIG=iglue_test_tasks_boxes36
TRTASK=XVNLI${LANG}_${SHOT}
TETASK=XVNLI${LANG}
TEXT_PATH=data/XVNLI/annotations/${LANG}/test.jsonl

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
PRETRAINED=checkpoints/iglue/few_shot/xvnli/${TRTASK}/${MODEL}/${best_lr}/XVNLI_${MODEL_CONFIG}/pytorch_model_best.bin
OUTPUT_DIR=results/iglue/few_shot/xvnli/${MODEL}/${best_lr}/${TRTASK}_${MODEL_CONFIG}/$TETASK/test

python eval_task.py \
  --bert_model huggingface/xlm-roberta-base --config_file config/${MODEL_CONFIG}.json \
  --from_pretrained ${PRETRAINED} \
  --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
  --split test --val_annotations_jsonpath ${TEXT_PATH} \
  --output_dir ${OUTPUT_DIR} --val_annotations_jsonpath ${TEXT_PATH} \

deactivate
