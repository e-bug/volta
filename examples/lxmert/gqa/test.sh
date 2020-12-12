#!/bin/bash

TASK=15
ANNOS=datasets/gqa/annotations
MODEL=lxmert
MODEL_CONFIG=lxmert
TASKS_CONFIG=lxmert_test_tasks
PRETRAINED=checkpoints/gqa/${MODEL}/GQA_${MODEL_CONFIG}/pytorch_model_5.bin
OUTPUT_DIR=results/gqa/${MODEL}

source activate volta

cd ../../..
python eval_task.py \
	--bert_model bert-base-uncased --config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
	--tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
	--output_dir ${OUTPUT_DIR}

python scripts/GQA_score.py \
  --preds_file ${OUTPUT_DIR}/pytorch_model_5.bin-/test_result.json \
  --truth_file ${ANNOS}/testdev_balanced_questions.json

conda deactivate
