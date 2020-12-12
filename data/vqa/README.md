# VQAv2

## Download
Download annotations from [here](https://sid.erda.dk/sharelink/dBf0YCAZaD).
Here, answers (`*_target.pkl`) are pickle versions of the VQA data redistributed by [LXMERT](https://github.com/airsplay/lxmert#vqa). 

The images used in this task are from MS COCO.
Check out [`mscoco`](../mscoco) for more details.

## Extract Image Features
Follow the procedure detailed in [`mscoco`](../mscoco).

## Serialize Image Features
Follow the procedure detailed in [`mscoco`](../mscoco).

---

The corpus directory looks as follows:
```text
vqa/
 |-- annotations/
 |    |-- train_target.pkl
 |    |-- trainval_ans2label.pkl
 |    |-- trainval_label2ans.pkl
 |    |-- v2_OpenEnded_mscoco_test2015_questions.json
 |    |-- v2_OpenEnded_mscoco_test-dev2015_questions.json
 |    |-- v2_OpenEnded_mscoco_train2014_questions.json
 |    |-- v2_OpenEnded_mscoco_val2014_questions.json
 |    |-- val_target.pkl

```
