#!/bin/bash

PATH=/home/bugliarello.e/data/nlvr2
cd $PATH

mkdir -p annotations
wget https://github.com/lil-lab/nlvr/raw/master/nlvr2/data/train.json
wget https://github.com/lil-lab/nlvr/raw/master/nlvr2/data/dev.json
wget https://github.com/lil-lab/nlvr/raw/master/nlvr2/data/test1.json
mv test1.json test.json