# Flickr30K

## Download
Fill in [this form](https://forms.illinois.edu/sec/229675) to request access to the Flickr30k images.
For more details, see the [dataset webpage](http://shannon.cs.illinois.edu/DenotationGraph/).

Run `download_captions.sh` setting the directory in which you want to store your corpus.

Hard negatives can be downloaded with [this link](https://sid.erda.dk/share_redirect/fDmTwg8szQ). 

## Extract Image Features
From the `volta` repository root, first use `airsplay/bottom-up-attention` to extract image features with Faster R-CNN. 
For example:
```text
# Path to training images
IMG_DIR=/home/bugliarello.e/data/flickr30k

docker run \
    --gpus all \
    -v $IMG_DIR:/workspace/images:ro \
    -v $(pwd)/data/flickr30k:/workspace/features \
    -v $(pwd)/data/snap:/workspace/snap:ro \
    --rm -it airsplay/bottom-up-attention bash

# Inside the docker container
pip install python-dateutil==2.5.0
cd features/
CUDA_VISIBLE_DEVICES=0 python extract_flickr30k_image.py --split flickr30k --group_id 0 --total_group 8
```
In this example, as we have 8 groups, the code above will need to be run 8 times, with `group_id` ranging from 0 to 7. 

NB. This script will extract features in this folder.
If you want to extract them in a different directory, make a copy of these two files in that directory,
and change `$(pwd)/data/flickr30k` in the docker execution appropriately.

Then, merge all the features extracted from the different processes:
```text
DATA_DIR=/home/bugliarello.e/data/flickr30k/resnet101_faster_rcnn_genome_imgfeats

python merge_flickr30k_image.py --split flickr30k --total_group 8 --datadir $DATA_DIR
```

## Serialize Image Features & Extract Captions
Finally, serialize the image features and extract their captions:
```text
bash convert_flickr30k_lmdb.sh
bash extract_captions.sh
```

---

The corpus directory looks as follows:
```text
flickr30k/
 |-- annotations/
 |    |-- test_ann.jsonl
 |    |-- train_ann.jsonl
 |    |-- valid_ann.jsonl
 |
 |-- dataset_flickr30k.json
 |-- hard_negative.pkl
 |
 |-- images/
 |
 |-- resnet101_faster_rcnn_genome_imgfeats/
 |    |-- test_obj36-36.tsv
 |    |-- train_obj36-36.tsv
 |    |-- valid_obj36-36.tsv
 |    |-- volta/
 |    |    |-- flickr30k_feat.lmdb

```
