#!/bin/bash

TASK=12
MODEL=lxmert
MODEL_CONFIG=lxmert
TASKS_CONFIG=lxmert_test_tasks
PRETRAINED=checkpoints/nlvr2/${MODEL}/NLVR2_${MODEL_CONFIG}/pytorch_model_19.bin
OUTPUT_DIR=results/nlvr2/${MODEL}

source activate volta

cd ../../..
python eval_task.py \
	--bert_model bert-base-uncased --config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
	--tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
	--output_dir ${OUTPUT_DIR}

conda deactivate