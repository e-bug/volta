#!/bin/bash

PATH=/home/bugliarello.e/data/refcoco_unc

source activate volta

python convert_refcoco_unc_dets36_lmdb.py --indir ${PATH}/imgfeats --outdir ${PATH}/imgfeats/volta

conda deactivate
