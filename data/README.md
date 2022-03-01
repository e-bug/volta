# Data Setup

We provide access to our preprocessed data (including extracted features) 
and preprocessing scripts to replicate our setup.

## Preprocessed Data

- [Conceptual Captions](https://sid.erda.dk/sharelink/hPGl0eTED3)
- [Flickr30k](https://sid.erda.dk/sharelink/CrLpUMgIKh)
- [GQA](https://sid.erda.dk/sharelink/erYmVzgpny)
- [MS COCO](https://sid.erda.dk/sharelink/e4iGIc3xYv)
- [NLVR2](https://sid.erda.dk/sharelink/fEZT4BVb9l)
- [RefCOCO (UNC)](https://sid.erda.dk/sharelink/GdBuBzki8m)
- [RefCOCO+ (UNC)](https://sid.erda.dk/sharelink/eEqyrN1IVs)
- [RefCOCOg (UMD)](https://sid.erda.dk/sharelink/EPpqybot4p)
- [SNLI-VE](https://sid.erda.dk/sharelink/g23Gqj9cad)
- [VQAv2](https://sid.erda.dk/sharelink/gzyTWulKAa)

More recent (from [IGLUE](https://github.com/e-bug/iglue)) and with more backbones:
- [Flickr30K](https://sid.erda.dk/sharelink/aW8MWVSlK1)
- [GQA](https://sid.erda.dk/sharelink/FtoWxwitOz)
- [MaRVL zero-shot](https://sid.erda.dk/sharelink/GYPEryxpVk) | [MaRVL few-shot](https://sid.erda.dk/sharelink/fMNmRmJgQA)
- [NLVR2](https://sid.erda.dk/sharelink/FjJUsFbRWO)
- [xFlickr&CO](https://sid.erda.dk/sharelink/cCObmVenjI)
- [WIT](https://sid.erda.dk/sharelink/escPrWm3Tt)

NB: I have noticed that uploading LMDB files made their size grow to the order of TBs.
So, instead, I recently uploaded the H5 versions that can quickly be converted to LMDB locally using [this script](https://github.com/e-bug/volta/blob/main/features_extraction/h5_to_lmdb.py).

## Preprocessing Steps

I originally relied on Hao Tan's [`airsplay/bottom-up-attention`](https://github.com/airsplay/bottom-up-attention) Docker image to extract image features from Faster R-CNN. 
For more details about the Docker image, see the [LXMERT repository](https://github.com/airsplay/lxmert#faster-r-cnn-feature-extraction).

Recently, I have switched to Hao Tan's [Detectron2 implementation](https://github.com/airsplay/py-bottom-up-attention) of 'Bottom-up feature extractor', which is compatible with the original Caffe implementation.
See [here](https://github.com/e-bug/volta/tree/main/features_extraction#resnet-101-backbone-36-boxes) for step-by-step instructions.

Moreover, it is possible to extract Faster R-CNN features with a ResNeXt-101 backbone from the [`mmf`](https://github.com/facebookresearch/mmf/) repository following [these instructions](https://github.com/e-bug/volta/tree/main/features_extraction#resnext-101-backbone-10-100-boxes).

---

For detailed preprocessing procedures, check out the README files for each data set in this folder or under [`feature_extraction/`](../feature_extraction).
