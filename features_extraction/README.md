# Features Extraction

## ResNet-101 backbone, 36 boxes

We use [Hao Tan's Detectron2 implementation of 'Bottom-up feature extractor'](https://github.com/airsplay/py-bottom-up-attention), which is compatible with [the original Caffe implementation](https://github.com/peteanderson80/bottom-up-attention).

Following LXMERT, we use the feature extractor which outputs 36 boxes per image.
We store features in hdf5 format.

### Install Feature Extractor

```bash
conda create -n py-bottomup python=3.6
source activate py-bottomup

cd py-bottom-up-attention

# Install python libraries
pip install -r requirements.txt
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Install detectron2
python setup.py build develop

# Download model
mkdir -p $HOME/.torch/fvcore_cache/models/
wget -P $HOME/.torch/fvcore_cache/models/ http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl

conda deactivate
```

NB:
1. Depending on your system, you might need to re-install PyTorch (e.g. I did `pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html`)
2. The model requires a newer version of gcc (I load it in my cluster as `source scl_source enable devtoolset-7`)
3. I had to also include the lib path (e.g. `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/pmh864/libs/anaconda3/lib python wit-test_boxes36_h5-proposal.py`)


### Extract features
```bash
# Flickr30K
python flickr30k_boxes36_h5-proposal.py \
    --root $INDIR \
    --outdir $OUTDIR
```


## ResNeXt-101 backbone, 10-100 boxes
Following M3P instructions, we use [`mmf`](https://github.com/facebookresearch/mmf/) to extract these features. 

### Install Feature Extractor
For installation, you need to be in a GPU node and with GCC >= 4.9.

```bash
conda create -n maskrcnn_mmf python=3.7
cd mmf
pip install --editable .

conda install ipython
pip install ninja yacs cython matplotlib
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-nightly
cd github/cocoapi/PythonAPI
python setup.py build_ext install
cd ../../vqa-maskrcnn-benchmark
python setup.py build develop
pip install opencv-python
```

### Extract features
```bash
# Flickr30K
cd mmf
python tools/scripts/features/extract_features_vmb.py \
    --model_name=X-101 \
    --image_dir=$INDIR \
    --output_folder=$OUTDIR
```
