#!/bin/bash

TASK=13
MODEL=vilbert
MODEL_CONFIG=vilbert_base
TASKS_CONFIG=vilbert_test_tasks
PRETRAINED=checkpoints/${MODEL}/VisualEntailment_${MODEL_CONFIG}/pytorch_model_2.bin
OUTPUT_DIR=results/snli_ve/${MODEL}

source activate volta

cd ../../..
python eval_task.py \
	--bert_model bert-base-uncased --config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
	--tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
	--output_dir ${OUTPUT_DIR}

conda deactivate
