#!/bin/bash

basedir=/home/projects/ku_00062

H5="${basedir}/data/gqa/features/vg-gqa_boxes36.h5"
LMDB="${basedir}/data/gqa/features/gqa-train_boxes36.lmdb"
TEXT="${basedir}/data/gqa/annotations/train_target.pkl"
rm -r $LMDB

source ${basedir}/envs/iglue/bin/activate

python gqa_boxes36_h5-to-tdlmdb.py --h5 $H5 --lmdb $LMDB --annotation $TEXT

deactivate
