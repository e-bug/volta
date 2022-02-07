#!/bin/bash

DATA="/home/projects/ku_00062/data/flickr30k/features"
H5="${DATA}/flickr30k_boxes36.h5"
LMDB="${DATA}/flickr30k_boxes36.lmdb"

source /home/projects/ku_00062/envs/iglue/bin/activate

cd ../..
python h5_to_lmdb.py --h5 $H5 --lmdb $LMDB

deactivate
