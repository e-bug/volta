# Bottom-up Attention with Detectron2 

The detectron2 system with **exactly the same model and weight** as the Caffe VG Faster R-CNN provided in [bottom-up-attetion](https://github.com/peteanderson80/bottom-up-attention).

The original [bottom-up-attetion](https://github.com/peteanderson80/bottom-up-attention) is implemented based on [Caffe](https://github.com/BVLC/caffe), which is not easy to install and is inconsistent with the training code in PyTorch.
Our project thus transfers the weights and models to [detectron2](https://github.com/facebookresearch/detectron2) that could be few-line installed and has PyTorch front-end.

The features extracted from this repo is compatible with LXMERT code and pre-trained models [here](https://github.com/airsplay/lxmert). Results have been locally verified.


## Installation
```
git clone https://github.com/airsplay/py-bottom-up-attention.git
cd py-bottom-up-attention

# Install python libraries
pip install -r requirements.txt
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Install detectron2
python setup.py build develop

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop

# or, as an alternative to `setup.py`, do
# pip install [--editable] .
```

## Demos

### Object Detection
[demo vg detection](demo/demo_vg_detection.ipynb)


### Feature Extraction 
With Attributes:
1. Single image: [demo extraction](demo/demo_feature_extraction_attr.ipynb)
2. Single image (Given boxes): [demo extraction](demo/demo_feature_extraction_attr_given_box.ipynb)

Without Attributes:
1. Single image: [demo extraction](demo/demo_feature_extraction.ipynb)
2. Single image (Given boxes): [demo extraction](demo/demo_feature_extraction_given_box.ipynb)

## Feature Extraction Scripts for MS COCO
**Note: this script does not include attribute. If you want to use attributes, please modify it according to [the demo](demo/demo_feature_extraction_attr.ipynb)**
1. For MS COCO (VQA): [vqa script](demo/detectron2_mscoco_proposal_maxnms.py)


## Note
1. The default weight is same to the 'alternative pretrained model' in the original github [here](https://github.com/peteanderson80/bottom-up-attention#demo), which is trained with 36 bbxes. If you want to use the original detetion trained with 10~100 bbxes, please use the following weight:
   ```
   http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr_original.pkl
   ```
2. The coordinate generated from the code is (x_left_corner, y_top_corner, x_right_corner, y_bottom_corner). Here is a visualization. Suppose the `box = [x0, y0, x1, y1]`, it annotates an RoI of:
   ```
   0-------------------------------------
    |                                   |
    y0 box[1]   |-----------|           |
    |           |           |           |
    |           |  Object   |           |
    y1 box[3]   |-----------|           |
    |                                   |
   H----------x0 box[0]-----x1 box[2]----
    0                                   W
   ```
3. If the link breaks, please contact me (at airsplay@cs.unc.edu) directly and I will share you the weight.

## External Links
1. The orignal CAFFE implementation [https://github.com/peteanderson80/bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention), and its [docker image](https://hub.docker.com/r/airsplay/bottom-up-attention).
2. [bottom-up-attention.pytorch](https://github.com/MILVLG/bottom-up-attention.pytorch) maintained by [MIL-LAB](http://mil.hdu.edu.cn/). 


## Proof of Correctness
1. As shown in [demo](demo/feature_correctness.ipynb)

Note: You might find a little difference between the caffe features and pytorch features in this verification demo. It is because the verification uses the setup "Given box" instead of "Predicted boxes". If the features are extracted from scratch (i.e., features with predicted boxes), they are exactly the same. 

Detailed explanation is here; "Given box" will use feature with the final predicted boxes (after box regression), however, the extracted features will use the features of the proposals. I illustrate this in below:

Feature extraction (using predicted boxes):
```
ResNet --> RPN --> RoiPooling + Res5 --> Box Regression --> BOX
                                      |-------------------> Feature --> Label
                                                                  |-> Attribute
```
Feature extraction (using given boxes):
```
ResNet --> RPN --> RoiPooling + Res5 --> Box Regression --> BOX
                                           |--> RoIPooling + Res5 --> Feature --> Label
                                                                              |-> Attribute
```

## Acknowledgement
The Caffe2PyTorch conversion code (not released here) is based on [Ruotian Luo](https://ttic.uchicago.edu/~rluo/)'s [PyTorch-ResNet](https://github.com/ruotianluo/pytorch-resnet) project. The project also refers to [Ross Girshick](https://www.rossgirshick.info/)'s old [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) on its way.



## References

Detectron2:
```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```

Bottom-up Attention:
```BibTeX
@inproceedings{Anderson2017up-down,
  author = {Peter Anderson and Xiaodong He and Chris Buehler and Damien Teney and Mark Johnson and Stephen Gould and Lei Zhang},
  title = {Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering},
  booktitle={CVPR},
  year = {2018}
}
```

LXMERT:
```BibTeX
@inproceedings{tan2019lxmert,
  title={LXMERT: Learning Cross-Modality Encoder Representations from Transformers},
  author={Tan, Hao and Bansal, Mohit},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year={2019}
}
```

