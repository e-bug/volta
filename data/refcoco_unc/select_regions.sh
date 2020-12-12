#!/bin/bash

PATH=/home/bugliarello.e/data/refcoco_unc

python select_regions.py \
  --infile ${PATH}/detections/refcoco_unc/res101_coco_minus_refer_notime_dets.json \
  --outfile ${PATH}/detections/refcoco_unc/res101_coco_minus_refer_notime_dets_36.json \
  --max_regions 36
