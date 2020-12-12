#!/bin/bash

TASK=1
MODEL=lxmert
MODEL_CONFIG=lxmert
TASKS_CONFIG=lxmert_trainval_tasks
PRETRAINED=checkpoints/conceptual_captions/${MODEL}/${MODEL_CONFIG}/pytorch_model_19.bin
OUTPUT_DIR=checkpoints/vqa/${MODEL}
LOGGING_DIR=logs/vqa

source activate volta

cd ../../..
python train_task.py \
	--bert_model bert-base-uncased --config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
	--tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
	--adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.01 --warmup_proportion 0.1 --clip_grad_norm 5.0 \
	--output_dir ${OUTPUT_DIR} \
	--logdir ${LOGGING_DIR} \
#	--resume_file ${OUTPUT_DIR}/VQA_${MODEL_CONFIG}/pytorch_ckpt_latest.tar

conda deactivate
