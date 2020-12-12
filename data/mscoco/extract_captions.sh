#!/bin/bash

PATH=/home/bugliarello.e/data/mscoco

source activate vilbert

python extract_captions.py --split train --infile ${PATH}/dataset_coco.json --outdir ${PATH}/annotations
python extract_captions.py --split valid --infile ${PATH}/dataset_coco.json --outdir ${PATH}/annotations
python extract_captions.py --split test1k --infile ${PATH}/dataset_coco.json --outdir ${PATH}/annotations

conda deactivate
