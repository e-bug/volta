#!/bin/bash

TASK=12
SHOT=10
LANG=zh
MODEL=uc2
MODEL_CONFIG=uc2_base
TASKS_CONFIG=iglue_fewshot_tasks_boxes36
TRTASK=MaRVL${LANG}_${SHOT}
TEXT_TR=data/marvl/few_shot/annotations/marvl-${LANG}_train${SHOT}.jsonl
FEAT_TR=data/marvl/few_shot/features/marvl-${LANG}_fewshot_boxes36.lmdb
TEXT_TE=data/nlvr2/annotations/dev.json
PRETRAINED=checkpoints/iglue/zero_shot/marvl/${MODEL}/NLVR2_${MODEL_CONFIG}/pytorch_model_best.bin

here=$(pwd)

source envs/iglue/bin/activate

cd ../../../../../..

for lr in 1e-4 5e-5 1e-5; do
  OUTPUT_DIR=checkpoints/iglue/few_shot/marvl/${TRTASK}/${MODEL}/${lr}
  LOGGING_DIR=logs/iglue/few_shot/marvl/${TRTASK}/${lr}/${MODEL_CONFIG}
  python train_task.py \
    --bert_model huggingface/xlm-roberta-base --config_file config/${MODEL_CONFIG}.json \
    --from_pretrained ${PRETRAINED} \
    --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --num_epoch 20 \
    --train_split train --train_annotations_jsonpath $TEXT_TR --train_features_lmdbpath $FEAT_TR \
    --lr $lr --batch_size 16 --gradient_accumulation_steps 1 --num_workers 0 --save_every_num_epochs 5 \
    --eval_batch_size 32 --max_val_batches 15 \
    --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.0001 --warmup_proportion 0.1 --clip_grad_norm 1.0 \
    --output_dir ${OUTPUT_DIR} \
    --logdir ${LOGGING_DIR} \
    &> ${here}/train.${lr}.log
done

deactivate
