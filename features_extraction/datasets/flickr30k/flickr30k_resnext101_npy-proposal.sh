#!/bib/bash

basedir=/home/projects/ku_00062
INDIR=${basedir}/data/flickr30k/images/
OUTDIR=${basedir}/data/flickr30k/features/flickr30k_X101.npy

mkdir -p $OUTDIR

source ${basedir}/envs/maskrcnn_mmf/bin/activate

cd ../../mmf
python tools/scripts/features/extract_features_vmb.py \
    --model_name=X-101 \
    --image_dir=$INDIR \
    --output_folder=$OUTDIR \

deactivate
