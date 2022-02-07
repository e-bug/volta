#!/bin/bash

DATADIR="/science/image/nlp-datasets/emanuele/data/wit/image_pixels/"

source activate /science/image/nlp-datasets/emanuele/envs/py-bottomup

python wit-test_boxes36_h5-proposal.py --root $DATADIR

conda deactivate
