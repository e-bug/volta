#!/bin/bash

basedir=/science/image/nlp-datasets/emanuele
model_file=models/detectron_model.pth
config_file=models/detectron_model.yaml
images_folder=${basedir}/data/flickr30k/images
output_folder=${basedir}/data/flickr30k/X-101_test

mkdir -p $output_folder

source ${basedir}/envs/maskrcnn_benchmark/bin/activate

cd ../..
python extract_features_vmb.py \
        --model_file $model_file \
        --config_file $config_file \
        --image_dir $images_folder \
        --output_folder $output_folder

deactivate
