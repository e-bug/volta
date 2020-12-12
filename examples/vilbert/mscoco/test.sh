#!/bin/bash

TASK=7
MODEL=vilbert
MODEL_CONFIG=vilbert_base
TASKS_CONFIG=vilbert_test_tasks
PRETRAINED=checkpoints/mscoco/${MODEL}/RetrievalFlickr30k_${MODEL_CONFIG}/pytorch_model_18.bin
OUTPUT_DIR=results/mscoco/${MODEL}

source activate volta

cd ../../..
python eval_retrieval.py \
	--bert_model bert-base-uncased --config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
	--tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --split test --batch_size 1 \
	--output_dir ${OUTPUT_DIR}

conda deactivate
