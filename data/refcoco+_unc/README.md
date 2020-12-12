# RefCOCO+ (UNC)

## Download
Run `download_data.sh` setting the directory in which you want to store your corpus.
This will download the annotations as well as the regions detected by [MAttNet](https://github.com/lichengunc/MAttNet). 

The images used in this task are from the `train2014` split of MS COCO.
Check out [`mscoco`](../mscoco) for more details.

## Extract Image Features
As a few images have more than 36 detected regions, as a first step, select the top-36 regions:

```text
bash select_regions.sh
```

Then, from the `volta` repository root, use `airsplay/bottom-up-attention` to extract image features with Faster R-CNN:
```text
# Path to training images
IMG_DIR=/home/bugliarello.e/data/mscoco/images
DET_DIR=/home/bugliarello.e/data/refcoco+_unc/detections/refcoco+_unc

docker run \
    --gpus all \
    -v $IMG_DIR:/workspace/images:ro \
    -v $(pwd)/data/refcoco+_unc:/workspace/features \
    -v $(pwd)/data/snap:/workspace/snap:ro \
    -v $DET_DIR:/workspace/detections:ro \
    --rm -it airsplay/bottom-up-attention bash

# Inside the docker container
pip install python-dateutil==2.5.0
cd features/
CUDA_VISIBLE_DEVICES=0 python extract_refcoco+_unc_dets36_image.py --group_id 0 --total_group 8
```
In this example, as we have 8 groups, the code above will need to be run 8 times, with `group_id` ranging from 0 to 7. 

NB. This script will extract features in this folder.
If you want to extract them in a different directory, make a copy of these two files in that directory,
and change `$(pwd)/data/refcoco+_unc` in the docker execution appropriately.

Then, merge all the features extracted from the different processes:
```text
DATA_DIR=/home/bugliarello.e/data/refcoco+_unc/resnet101_faster_rcnn_genome_imgfeats

python merge_refcoco+_unc_dets36_image.py --total_group 8 --datadir $DATA_DIR
```

## Serialize Image Features
Finally, serialize the image features:
```text
bash convert_refcoco+_unc_dets36_lmdb.sh
```

---

The corpus directory looks as follows:
```text
refcoco+_unc/
 |-- annotations/
 |    |-- refcoco+/
 |    |    |-- instances.json
 |    |    |-- refs(unc).p
 |
 |-- detections/
 |    |-- refcoco+_unc/
 |    |    |-- res101_coco_minus_refer_notime_dets.json
 |    |    |-- res101_coco_minus_refer_notime_masks.json
 |    |    |-- res101_coco_minus_refer_notime_dets_36.json
 |
 |-- resnet101_faster_rcnn_genome_imgfeats/
 |    |-- refcoco+_unc_dets36_obj36-36.tsv
 |    |-- volta/
 |    |    |-- refcoco+_unc_dets36_feat.lmdb

```
