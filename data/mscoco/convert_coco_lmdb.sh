#!/bin/bash

PATH=/home/bugliarello.e/data/mscoco

source activate volta

python convert_coco_lmdb.py --split trainval --indir ${PATH}/imgfeats --outdir ${PATH}/imgfeats/volta
python convert_coco_lmdb.py --split test --indir ${PATH}/imgfeats --outdir ${PATH}/imgfeats/volta

conda deactivate
