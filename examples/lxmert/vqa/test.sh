#!/bin/bash

TASK=1
MODEL=lxmert
MODEL_CONFIG=lxmert
TASKS_CONFIG=lxmert_test_tasks
PRETRAINED=checkpoints/${MODEL}/VQA_${MODEL_CONFIG}/pytorch_model_19.bin
OUTPUT_DIR=results/vqa/${MODEL}

source activate volta

cd ../../..
python eval_task.py \
	--config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
	--tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
	--output_dir ${OUTPUT_DIR}

conda deactivate
