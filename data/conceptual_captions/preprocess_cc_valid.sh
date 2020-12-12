#!/bin/bash

CORPUS="/data/bugliarello.e/conceptual_captions"

source activate volta

python preprocess_cc_valid.py $CORPUS

conda deactivate
