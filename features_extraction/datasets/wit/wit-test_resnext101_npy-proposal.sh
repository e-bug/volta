#!/bib/bash

basedir=/science/image/nlp-datasets/emanuele/data/wit/image_pixels
INDIR=${basedir}
OUTDIR=${basedir}/wit_test_resnext101

mkdir -p $OUTDIR

source activate /science/image/nlp-datasets/emanuele/envs/maskrcnn_mmf

cd ../../mmf
python tools/scripts/features/extract_features_vmb_wit-test.py \
    --model_name=X-101 \
    --image_dir=$INDIR \
    --output_folder=$OUTDIR \

conda deactivate
