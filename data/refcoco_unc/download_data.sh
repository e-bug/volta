#!/bin/bash

PATH=/home/bugliarello.e/data/refcoco+_unc
cd $PATH

# Captions
mkdir -p annotations
cd annotations
wget http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
unzip refcoco+.zip
cd ..

# MattNet detections
wget http://bvision.cs.unc.edu/licheng/MattNet/detections.zip
unzip detections.zip
rm -r detections/refcoco_unc detections/refcocog_umd
