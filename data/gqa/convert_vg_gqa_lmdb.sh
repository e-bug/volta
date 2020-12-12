#!/bin/bash

PATH=/home/bugliarello.e/data/gqa

source activate volta

python convert_vg_gqa_lmdb.py --indir ${PATH}/imgfeats --outdir ${PATH}/imgfeats/volta

conda deactivate
