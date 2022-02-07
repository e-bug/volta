#!/bib/bash

basedir=/home/projects/ku_00062

mkdir -p $OUTDIR

source ${basedir}/envs/py-bottomup/bin/activate

for lang in id sw ta tr zh; do
  INDIR=${basedir}/data/marvl/images/${lang}/all
  OUTDIR=${basedir}/data/marvl/features/
  
  python marvl_boxes36_h5-proposal.py \
    --root $INDIR \
    --outdir $OUTDIR \
    --lang $lang

done

deactivate
