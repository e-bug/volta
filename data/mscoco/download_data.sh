#!/bin/bash

PATH=/home/bugliarello.e/data/mscoco
cd $PATH

# Captions
wget https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip caption_datasets.zip
rm dataset_flickr8k.json dataset_flickr30k.json

# Images
mkdir -p images
cd images

wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/test2015.zip

unzip train2014.zip
unzip val2014.zip
unzip test2015.zip
