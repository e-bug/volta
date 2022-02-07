#!/bin/bash

indir="/science/image/nlp-datasets/emanuele/data/wit/features"
outdir="/science/image/nlp-datasets/emanuele/data/wit/wit-en_train_boxes36.lmdb"
ids="/science/image/nlp-datasets/emanuele/data/wit/train_en_ids.lst"

rm -r $outdir

source activate /science/image/nlp-datasets/emanuele/envs/volta

python wit-trainval_boxes36_h5-to-tdlmdb.py --h5_dir $indir --lmdb $outdir --ids_fname $ids --num_imgs 500000

conda deactivate
