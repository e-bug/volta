#!/bib/bash

basedir=/home/projects/ku_00062
INDIR=${basedir}/data/flickr30k/images/
OUTDIR=${basedir}/data/flickr30k/features/

mkdir -p $OUTDIR

source ${basedir}/envs/py-bottomup/bin/activate

python flickr30k_boxes36_h5-proposal.py \
  --root $INDIR \
  --outdir $OUTDIR \

deactivate
