# VOLTA: Visiolinguistic Transformer Architectures

This is the implementation of the framework described in the paper:
> Emanuele Bugliarello, Ryan Cotterell, Naoaki Okazaki and Desmond Elliott. [Multimodal Pretraining Unmasked: Unifying the Vision and Language BERTs](https://arxiv.org/abs/2011.15124). arXiv preprint arXiv:2011.15124, November 2020.

We provide the code for reproducing our results, preprocessed data and pretrained models.


## Repository Setup

You can clone this repository with submodules included issuing: <br>
`git clone git@github.com:e-bug/volta`

1\. Create a fresh conda environment, and install all dependencies.
```text
conda create -n volta python=3.6
conda activate volta
pip install -r requirements.txt
```

2\. Install PyTorch
```text
conda install pytorch=1.4.0 torchvision=0.5 cudatoolkit=10.1 -c pytorch
```

3\. Install [apex](https://github.com/NVIDIA/apex).
If you use a cluster, you may want to first run commands like the following:
```text
module load cuda/10.1.105
module load gcc/8.3.0-cuda
```

4\. Setup the `refer` submodule for Referring Expression Comprehension:
```
cd tools/refer; make
```

5\. Install this codebase as a package in this environment.
```text
python setup.py develop
```

## Data

Check out [`data/README.md`](data/README.md) for links to preprocessed data and data preparation steps. 


## Models

Check out [`MODELS.md`](MODELS.md) for links to pretrained models and how to define new ones in VOLTA.

Model configuration files are stored in [config/](config). 


## Training and Evaluation

We provide sample scripts to train (i.e. pretrain or fine-tune) and evaluate models in [examples/](examples).
These include ViLBERT, LXMERT and VL-BERT as detailed in the original papers, 
as well as ViLBERT, LXMERT, VL-BERT, VisualBERT and UNITER as specified in our controlled study.

Task configuration files are stored in [config_tasks/](config_tasks).


## License

This work is licensed under the MIT license. See [`LICENSE`](LICENSE) for details. 
Third-party software and data sets are subject to their respective licenses. <br>
If you find our code/data/models or ideas useful in your research, please consider citing the paper:
```
@article{bugliarello-etal-2020-multimodal,
    title = "Multimodal Pretraining Unmasked: {U}nifying the Vision and Language {BERT}s",
    author = "Bugliarello, Emanuele  and
      Cotterell, Ryan and
      Okazaki, Naoaki and
      Elliott, Desmond",
    journal = "arXiv preprint arXiv:2011.15124"
    year = "2020",
    url = "https://arxiv.org/abs/2011.15124",
}
```


## Acknowledgement

Our codebase heavily relies on these excellent repositories:
- [vilbert-multi-task](https://github.com/facebookresearch/vilbert-multi-task)
- [vilbert_beta](https://github.com/jiasenlu/vilbert_beta)
- [lxmert](https://github.com/airsplay/lxmert)
- [VL-BERT](https://github.com/jackroos/VL-BERT)
- [visualbert](https://github.com/uclanlp/visualbert)
- [UNITER](https://github.com/ChenRocks/UNITER)
- [pytorch-transformers](https://github.com/huggingface/pytorch-transformers)
- [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention)

