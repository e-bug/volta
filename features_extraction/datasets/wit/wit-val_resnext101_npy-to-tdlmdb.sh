#!/bin/bash

indir="/science/image/nlp-datasets/emanuele/data/wit/features"
outdir="/science/image/nlp-datasets/emanuele/data/wit/features/wit-en_val_X101.lmdb"
ids="/science/image/nlp-datasets/emanuele/data/wit/val_en_ids.lst"

rm -r $outdir

python wit-trainval_resnext101_npy-to-tdlmdb.py --features_dir $indir --lmdb $outdir --ids_fname $ids --num_imgs 4632
