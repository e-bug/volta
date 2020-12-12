import json
import tqdm
import argparse


def evaluate(preds_list, truth_dict):
    score = 0.
    for entry in tqdm.tqdm(preds_list, total=len(preds_list)):
        quesid = entry["questionId"]
        pred = entry["prediction"]
        label = truth_dict[quesid]["answer"]
        if pred in label:
            score += 1.
    return score / len(preds_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_file", default="", type=str, help="Path to predictions file")
    parser.add_argument("--truth_file", default="", type=str, help="Path to ground truth file")
    args = parser.parse_args()

    preds_list = json.load(open(args.preds_file))
    truth_dict = json.load(open(args.truth_file))
    
    score = evaluate(preds_list, truth_dict)
    print(100*score)
