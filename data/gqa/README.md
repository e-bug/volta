# GQA

## Download
Download image files from the [GQA website](https://cs.stanford.edu/people/dorarad/gqa/download.html) and extract them in the corpus directory.

Annotations can be downloaded with [this link](https://sid.erda.dk/sharelink/AE76y8ThUK).
These annotations are the re-distributed json files for the GQA balanced version dataset from [LXMERT](https://github.com/airsplay/lxmert#gqa), and then preprocessed for VOLTA/ViLBERT.

## Extract Image Features
From the `volta` repository root, first use `airsplay/bottom-up-attention` to extract image features with Faster R-CNN. 
For example:
```text
# Path to training images
IMG_DIR=/home/bugliarello.e/data/gqa/images

docker run \
    --gpus all \
    -v $IMG_DIR:/workspace/images:ro \
    -v $(pwd)/data/gqa:/workspace/features \
    -v $(pwd)/data/snap:/workspace/snap:ro \
    --rm -it airsplay/bottom-up-attention bash

# Inside the docker container
pip install python-dateutil==2.5.0
cd features/
CUDA_VISIBLE_DEVICES=0 python extract_vg_gqa_image.py --group_id 0 --total_group 8
```
In this example, as we have 8 groups, the code above will need to be run 8 times, with `group_id` ranging from 0 to 7. 

NB. This script will extract features in this folder.
If you want to extract them in a different directory, make a copy of these two files in that directory,
and change `$(pwd)/data/gqa` in the docker execution appropriately.

Then, merge all the features extracted from the different processes:
```text
DATA_DIR=/home/bugliarello.e/data/gqa/resnet101_faster_rcnn_genome_imgfeats

python merge_vg_gqa_image.py --total_group 8 --datadir $DATA_DIR
```

## Serialize Image Features
Finally, serialize the image features:
```text
bash convert_vg_gqa_lmdb.sh
```

---

The corpus directory looks as follows:
```text
gqa/
 |-- annotations/
 |    |-- testdev_balanced_questions.json
 |    |-- train_target.pkl
 |    |-- trainval_ans2label.pkl
 |    |-- trainval_label2ans.pkl
 |    |-- trainval_target.pkl
 |    |-- val_target.pkl
 |
 |-- images/
 |
 |-- resnet101_faster_rcnn_genome_imgfeats/
 |    |-- vg_gqa_obj36-36.tsv
 |    |-- volta/
 |    |    |-- vg_gqa_feat.lmdb

```
