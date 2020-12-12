#!/bin/bash

PATH=/home/bugliarello.e/data/refcocog_umd

source activate volta

python convert_refcocog_umd_dets36_lmdb.py --indir ${PATH}/imgfeats --outdir ${PATH}/imgfeats/volta

conda deactivate
