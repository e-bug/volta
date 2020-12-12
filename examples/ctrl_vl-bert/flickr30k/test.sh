#!/bin/bash

TASK=8
MODEL=ctrl_vl-bert
MODEL_CONFIG=ctrl_vl-bert_base
TASKS_CONFIG=ctrl_test_tasks
PRETRAINED=checkpoints/flickr30k/${MODEL}/RetrievalFlickr30k_${MODEL_CONFIG}/pytorch_model_17.bin
OUTPUT_DIR=results/flickr30k/${MODEL}

source activate volta

cd ../../..
python eval_retrieval.py \
	--bert_model bert-base-uncased --config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
	--tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --split test --batch_size 1 \
	--output_dir ${OUTPUT_DIR}

conda deactivate
