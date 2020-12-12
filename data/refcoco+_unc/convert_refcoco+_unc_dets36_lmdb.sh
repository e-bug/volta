#!/bin/bash

PATH=/home/bugliarello.e/data/refcoco+_unc

source activate volta

python convert_refcoco+_unc_dets36_lmdb.py --indir ${PATH}/imgfeats --outdir ${PATH}/imgfeats/volta

conda deactivate
