#!/bib/bash

basedir=/home/projects/ku_00062
INDIR=${basedir}/data/nlvr2/images/
OUTDIR=${basedir}/data/nlvr2/features/

mkdir -p $OUTDIR

source ${basedir}/envs/py-bottomup/bin/activate

python nlvr2_boxes36_h5-proposal.py \
  --root $INDIR \
  --outdir $OUTDIR \

conda deactivate
