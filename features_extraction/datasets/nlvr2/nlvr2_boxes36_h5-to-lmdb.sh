#!/bin/bash

DATA="/home/projects/ku_00062/data/nlvr2/features"
H5="${DATA}/nlvr2_boxes36.h5"
LMDB="${DATA}/nlvr2_boxes36.lmdb"

source /home/projects/ku_00062/envs/iglue/bin/activate

cd ../..
python h5_to_lmdb.py --h5 $H5 --lmdb $LMDB

deactivate
