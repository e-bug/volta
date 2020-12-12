#!/bin/bash

TASK=1
MODEL=vl-bert
MODEL_CONFIG=vl-bert_base
TASKS_CONFIG=vl-bert_test_tasks
PRETRAINED=checkpoints/${MODEL}/VQA_${MODEL_CONFIG}/pytorch_model_18.bin
OUTPUT_DIR=results/vqa/${MODEL}

source activate volta

cd ../../..
python eval_task.py \
	--bert_model bert-base-uncased --config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
	--tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
	--output_dir ${OUTPUT_DIR}

conda deactivate
