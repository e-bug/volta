#!/bib/bash

basedir=/home/projects/ku_00062
OUTDIR=${basedir}/data/marvl/few_shot/features/

mkdir -p $OUTDIR

source ${basedir}/envs/py-bottomup/bin/activate

for lang in id sw ta tr zh; do
  INDIR=${basedir}/data/marvl/few_shot/images/${lang}/all

  python marvl_boxes36_h5-proposal.py \
    --root $INDIR \
    --outdir $OUTDIR \
    --lang $lang

done

deactivate
