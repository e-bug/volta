#!/bin/bash

PATH=/home/bugliarello.e/data/flickr30k

source activate volta

python convert_flickr30k_lmdb.py --split flickr30k --indir ${PATH}/imgfeats --outdir ${PATH}/imgfeats/volta

conda deactivate
