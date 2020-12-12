#!/bin/bash

TASK=10
MODEL=ctrl_lxmert
MODEL_CONFIG=ctrl_lxmert
TASKS_CONFIG=ctrl_test_tasks
PRETRAINED=checkpoints/refcoco+_unc/${MODEL}/refcoco+_${MODEL_CONFIG}/pytorch_model_18.bin
OUTPUT_DIR=results/refcoco+_unc/${MODEL}

source activate volta

cd ../../..
python eval_task.py \
	--bert_model bert-base-uncased --config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
	--tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
	--output_dir ${OUTPUT_DIR}

conda deactivate
