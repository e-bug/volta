#!/bin/bash

H5="/home/projects/ku_00062/data/wit/features/wit_test_boxes36.h5"
LMDB="/home/projects/ku_00062/data/wit/features/wit_test_boxes36.lmdb"

rm -r $LMDB

source /home/projects/ku_00062/envs/iglue/bin/activate

cd ../../..
python h5_to_lmdb.py --h5 $H5 --lmdb $LMDB

deactivate
