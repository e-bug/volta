#!/bin/bash

PATH=/home/bugliarello.e/data/flickr30k

source activate vilbert

python extract_captions.py --split train --infile ${PATH}/dataset_flickr30k.json --outdir ${PATH}/annotations
python extract_captions.py --split valid --infile ${PATH}/dataset_flickr30k.json --outdir ${PATH}/annotations
python extract_captions.py --split test --infile ${PATH}/dataset_flickr30k.json --outdir ${PATH}/annotations

conda deactivate
