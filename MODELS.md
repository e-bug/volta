# Models

## Pretrained Models

We distribute models pretrained on Conceptual Captions.
We share ViLBERT, LXMERT and VL-BERT pretrained as originally presented in their papers,
as well as the weights for ViLBERT, LXMERT, VL-BERT, VisualBERT and UNITER pretrained in our controlled setup.
For the latter, we distribute the weights that lead to higher average downstream performance when fine-tuned once.

| Model             | VQAv2 | RefCOCO+ | NLVR2 | Flickr30k IR | Flickr30k TR |
|-------------------|-------|----------|-------|--------------|--------------|
| [ViLBERT](https://sid.erda.dk/share_redirect/AgwrMiOjTv)           | 66.68 | 70.49    | 74.26 | 58.90        | 75.50        |
| [LXMERT](https://sid.erda.dk/share_redirect/fYBrp01t8M)            | 67.98 |          | 71.58 |              |              |
| [VL-BERT](https://sid.erda.dk/share_redirect/cCMQ8SXHdf)           | 67.44 | 71.00    |       |              |              |
| [ViLBERT (CTRL)](https://sid.erda.dk/share_redirect/aQCx8cLWK7)    | 68.97 | 70.53    | 72.24 | 60.34        | 78.80        |
| [LXMERT (CTRL)](https://sid.erda.dk/share_redirect/Dp1g16DIA5)     | 67.52 | 70.49    | 71.09 | 58.62        | 74.90        |
| [VL-BERT (CTRL)](https://sid.erda.dk/share_redirect/Dr8geMQyRd)    | 68.23 | 71.23    | 73.22 | 57.62        | 70.90        |
| [VisualBERT (CTRL)](https://sid.erda.dk/share_redirect/GCBlzUuoJl) | 69.03 | 70.02    | 72.70 | 61.48        | 75.20        |
| [UNITER (CTRL)](https://sid.erda.dk/share_redirect/FeYIWpMSFg)     | 68.67 | 71.45    | 73.73 | 60.54        | 76.40        |

### Checkpoints by Random Seed
All the models pretrained with 10 random seeds in our controlled setup can be downloaded from [here](https://sid.erda.dk/sharelink/GWj9Oh5dx4).

### Conversions of Original Models into VOLTA
| Model             | Source |
|-------------------|--------|
| [LXMERT (Original)](https://sid.erda.dk/share_redirect/cFGANaAtmN) | [airsplay/lxmert](https://nlp.cs.unc.edu/data/github_pretrain/lxmert20/Epoch20_LXRT.pth) |

### Multilingual Models

| Model      | XVNLI | xGQA | MaRVL | xFlickr&CO IR | xFlickr&CO TR | WIT IR | WIT TR |
|------------|-------|------|-------|---------------|---------------|--------|--------|
| [mUNITER](https://sid.erda.dk/sharelink/eYYhxHbth5)| 53.69 |  9.97 | 53.72 | 8.06 | 8.86 | 9.16 | 10.48 |
| [xUNITER](https://sid.erda.dk/sharelink/fhT5bmg56Q)| 58.48 | 21.72 | 54.59 | 14.04 | 13.51 | 8.72 | 9.81 |
| [UC2](https://sid.erda.dk/sharelink/gEHwACIX57)    | 62.05 | 29.35 | 57.28 | 20.31 | 17.89 | 7.83 | 9.09 |
| [M3P](https://sid.erda.dk/sharelink/hWHrRY7lag)    | 58.25 | 28.17 | 56.00 | 12.91 | 11.90 | 8.12 | 9.98 |


## Models Definition

Models are defined in configuration files (see [config/](config) for some examples).
Rather than using Transformer layers, we specify attention and feed-forward sub-layers for each modality, 
which allows to quickly extend proposed architectures.
In particular, the following sub-layers are defined:
- `tt_attn_sublayers`: text-text attention sub-layers
- `tv_attn_sublayers`: text-vision attention sub-layers (text used as query, vision as context)
- `vt_attn_sublayers`: vision-text attention sub-layers (vision used as query, text as context)
- `vv_attn_sublayers`: vision-vision attention sub-layers
- `t_ff_sublayers`: feed-forward sub-layers for the text modality
- `v_ff_sublayers`: feed-forward sub-layers for the vision modality

In addition, the following parameters allow to tune parameter sharing across modalities:
- `shared_sublayers`: sub-layers that share parameters between modalities 
- `single_ln_sublayers`: sub-layers in which text and vision tensors are concatenated and fed into a single LN layer

Finally, `bert_layer2attn_sublayer` and `bert_layer2ff_sublayer` are used to load text-only BERT layers into VOLTA ones. 

The following figure shows how these sub-layers are used to construct ViLBERT:
![](./ViLBERT_VOLTA.png)
