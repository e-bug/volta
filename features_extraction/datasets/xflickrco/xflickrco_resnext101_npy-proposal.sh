#!/bib/bash

basedir=/home/projects/ku_00062/
IMGDIR=${basedir}/data/xFlickrCO/images

source ${basedir}/envs/maskrcnn_mmf/bin/activate

cd ../../mmf
for split in test few; do
  INDIR=${IMGDIR}/${split}
  OUTDIR=${basedir}/data/xFlickrCO/features/xflickrco-${split}_X101.npy

  python tools/scripts/features/extract_features_vmb.py \
    --model_name=X-101 \
    --image_dir=$INDIR \
    --output_folder=$OUTDIR \

done

deactivate
