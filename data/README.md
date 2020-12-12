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


## Preprocessing Steps

We rely on the `airsplay/bottom-up-attention` Docker image to extract image features from Faster R-CNN.
This docker file for [`bottom-up-attention`](https://github.com/peteanderson80/bottom-up-attention) is available on 
[docker hub](https://hub.docker.com/r/airsplay/bottom-up-attention) and can be downloaded with:
```text
sudo docker pull airsplay/bottom-up-attention
```

For more details about the Docker image, 
see the [LXMERT repository](https://github.com/airsplay/lxmert#faster-r-cnn-feature-extraction).
Our scripts assume the [pretrained Caffe models](https://sid.erda.dk/sharelink/EgyY7wjCNf) 
to be stored under [snap/pretrained/](snap/pretrained).

---

Check out the README files for each data set for detailed preprocessing procedures. 
