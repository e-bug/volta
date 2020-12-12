#!/bin/bash

CORPUS="/data/bugliarello.e/conceptual_captions"

source activate volta

python preprocess_cc_train.py $CORPUS

conda deactivate
