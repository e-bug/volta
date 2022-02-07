#!/bib/bash

basedir=/home/projects/ku_00062

source ${basedir}/envs/maskrcnn_mmf/bin/activate

cd ../../mmf
for lang in id sw ta tr zh; do
  INDIR=${basedir}/data/marvl/images/${lang}/all
  OUTDIR=${basedir}/data/marvl/features/marvl-${lang}_X101.npy

  python tools/scripts/features/extract_features_vmb.py \
    --model_name=X-101 \
    --image_dir=$INDIR \
    --output_folder=$OUTDIR \

done

deactivate
