#!/bib/bash

basedir=/home/projects/ku_00062
INDIR=${basedir}/data/xFlickrCO/images/
OUTDIR=${basedir}/data/xFlickrCO/features/

mkdir -p $OUTDIR

source ${basedir}/envs/py-bottomup/bin/activate

python xflickrco_boxes36_h5-proposal.py \
  --root $INDIR \
  --outdir $OUTDIR \
  --split test

python xflickrco_boxes36_h5-proposal.py \
  --root $INDIR \
  --outdir $OUTDIR \
  --split few

deactivate
