#!/bin/bash

PATH=/home/bugliarello.e/data/refcocog_umd

python select_regions.py \
  --infile ${PATH}/detections/refcocog_umd/res101_coco_minus_refer_notime_dets.json \
  --outfile ${PATH}/detections/refcocog_umd/res101_coco_minus_refer_notime_dets_36.json \
  --max_regions 36
