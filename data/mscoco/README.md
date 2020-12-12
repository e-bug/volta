# MS COCO

## Download
Run `download_data.sh` setting the directory in which you want to store your corpus.

Hard negatives can be downloaded with [this link](https://sid.erda.dk/share_redirect/BLPh86CGIH).

## Extract Image Features
From the `volta` repository root, first use `airsplay/bottom-up-attention` to extract image features with Faster R-CNN. 
For example:
```text
# Path to training images
IMG_DIR=/home/bugliarello.e/data/mscoco/images

docker run \
    --gpus all \
    -v $IMG_DIR:/workspace/images:ro \
    -v $(pwd)/data/mscoco:/workspace/features \
    -v $(pwd)/data/snap:/workspace/snap:ro \
    --rm -it airsplay/bottom-up-attention bash

# Inside the docker container
pip install python-dateutil==2.5.0
cd features/
CUDA_VISIBLE_DEVICES=0 python extract_coco_image.py --split train --group_id 0 --total_group 8
```
In this example, as we have 8 groups, the code above will need to be run 8 times, with `group_id` ranging from 0 to 7. 

NB. This script will extract features in this folder.
If you want to extract them in a different directory, make a copy of these two files in that directory,
and change `$(pwd)/data/mscoco` in the docker execution appropriately.

Then, merge all the features extracted from the different processes:
```text
DATA_DIR=/home/bugliarello.e/data/mscoco/resnet101_faster_rcnn_genome_imgfeats

python merge_coco_image.py --split train --total_group 8 --datadir $DATA_DIR
```

## Serialize Image Features & Extract Captions
Finally, serialize the image features and extract their captions:
```text
bash convert_coco_lmdb.sh
bash extract_captions.sh
```

---

The corpus directory looks as follows:
```text
mscoco/
 |-- annotations/
 |    |-- test_ann.jsonl
 |    |-- train_ann.jsonl
 |    |-- valid_ann.jsonl
 |
 |-- dataset_coco.json
 |-- hard_negative.pkl
 |
 |-- images/
 |    |-- train2014
 |    |-- val2014
 |    |-- test2015
 |
 |-- resnet101_faster_rcnn_genome_imgfeats/
 |    |-- test_obj36-36.tsv
 |    |-- train_obj36-36.tsv
 |    |-- valid_obj36-36.tsv
 |    |-- volta/
 |    |    |-- trainval_feat.lmdb
 |    |    |-- test_feat.lmdb

```
