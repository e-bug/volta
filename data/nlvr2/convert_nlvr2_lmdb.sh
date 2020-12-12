#!/bin/bash

PATH=/home/bugliarello.e/data/nlvr2

source activate vilbert

python convert_nlvr2_lmdb.py --indir ${PATH}/imgfeats --outdir ${PATH}/imgfeats/volta

conda deactivate
