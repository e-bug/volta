#!/bib/bash

basedir=/home/projects/ku_00062
INDIR=${basedir}/data/nlvr2/images/
OUTDIR=${basedir}/data/nlvr2/features/nlvr2_X101.npy

mkdir -p $OUTDIR

source ${basedir}/envs/maskrcnn_mmf/bin/activate

cd ../../mmf
python tools/scripts/features/extract_features_vmb.py \
    --model_name=X-101 \
    --image_dir=$INDIR \
    --output_folder=$OUTDIR \

deactivate
