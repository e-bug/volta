#!/bin/bash

PATH=/home/bugliarello.e/data/flickr30k
cd $PATH

wget https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip caption_datasets.zip
rm dataset_coco.json dataset_flickr8k.json
