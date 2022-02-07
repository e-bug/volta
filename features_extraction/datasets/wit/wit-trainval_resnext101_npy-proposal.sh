#!/bib/bash

CHUNK=$1

basedir=/science/image/nlp-datasets/emanuele/data/wit/features
INDIR=${basedir}
OUTDIR=${basedir}/wit_${CHUNK}_resnext101

mkdir -p $OUTDIR

source activate /science/image/nlp-datasets/emanuele/envs/maskrcnn_mmf

cd ../../mmf
python tools/scripts/features/extract_features_vmb_wit.py \
    --model_name=X-101 \
    --image_dir=$INDIR \
    --output_folder=$OUTDIR \
    --chunk=$CHUNK \

conda deactivate
