# NLVR2

## Download
Download image files following the [NLVR repository](https://github.com/lil-lab/nlvr/tree/master/nlvr2#downloading-the-images) 
or by filling out [this form](https://goo.gl/forms/yS29stWnFWzrDBFH3).

Run `download_captions.sh` setting the directory in which you want to store your corpus.

## Extract Image Features
From the `volta` repository root, first use `airsplay/bottom-up-attention` to extract image features with Faster R-CNN. 
For example:
```text
# Path to training images
IMG_DIR=/home/bugliarello.e/data/nlvr2/images

docker run \
    --gpus all \
    -v $IMG_DIR:/workspace/images:ro \
    -v $(pwd)/data/nlvr2:/workspace/features \
    -v $(pwd)/data/snap:/workspace/snap:ro \
    --rm -it airsplay/bottom-up-attention bash

# Inside the docker container
pip install python-dateutil==2.5.0
cd features/
CUDA_VISIBLE_DEVICES=0 python extract_nlvr2_image.py --group_id 0 --total_group 8
```
In this example, as we have 8 groups, the code above will need to be run 8 times, with `group_id` ranging from 0 to 7. 

NB. This script will extract features in this folder.
If you want to extract them in a different directory, make a copy of these two files in that directory,
and change `$(pwd)/data/nlvr2` in the docker execution appropriately.

Then, merge all the features extracted from the different processes:
```text
DATA_DIR=/home/bugliarello.e/data/nlvr2/resnet101_faster_rcnn_genome_imgfeats

python merge_nlvr2_image.py --total_group 8 --datadir $DATA_DIR
```

## Serialize Image Features
Finally, serialize the image features:
```text
bash convert_nlvr2_lmdb.sh
```

---

The corpus directory looks as follows:
```text
nlvr2/
 |-- annotations/
 |    |-- dev.json
 |    |-- test.json
 |    |-- train.json
 |
 |-- images/
 |
 |-- resnet101_faster_rcnn_genome_imgfeats/
 |    |-- nlvr2_obj36-36.tsv
 |    |-- volta/
 |    |    |-- nlvr2_feat.lmdb

```
