#!/bin/bash

INDIR="/home/projects/ku_00062/data/wit/features/wit_test_resnext101.npy"
LMDB="/home/projects/ku_00062/data/wit/features/wit_test_X101.lmdb"
rm -r $LMDB

source /home/projects/ku_00062/envs/iglue/bin/activate

cd ../..
python npy_to_lmdb.py --mode convert --features_folder $INDIR --lmdb_path $LMDB

deactivate
