# Copyright (c) Facebook, Inc. and its affiliates.

# Requires vqa-maskrcnn-benchmark (https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark)
# to be built and installed. Category mapping for visual genome can be downloaded from
# https://dl.fbaipublicfiles.com/pythia/data/visual_genome_categories.json
# When the --background flag is set, the index saved with key "objects" in
# info_list will be +1 of the Visual Genome category mapping above and 0
# is the background class. When the --background flag is not set, the
# index saved with key "objects" in info list will match the Visual Genome
# category mapping.
import argparse
import os

import cv2
import base64
import numpy as np
import pandas as pd
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from mmf.utils.download import download
from PIL import Image

from tools.scripts.features.extraction_utils import chunks, get_image_files


class FeatureExtractor:

    MODEL_URL = {
        "X-101": "https://dl.fbaipublicfiles.com/pythia/"
        + "detectron_model/detectron_model.pth",
        "X-152": "https://dl.fbaipublicfiles.com/pythia/"
        + "detectron_model/detectron_model_x152.pth",
    }
    CONFIG_URL = {
        "X-101": "https://dl.fbaipublicfiles.com/pythia/"
        + "detectron_model/detectron_model.yaml",
        "X-152": "https://dl.fbaipublicfiles.com/pythia/"
        + "detectron_model/detectron_model_x152.yaml",
    }

    MAX_SIZE = 1333
    MIN_SIZE = 800

    def __init__(self):
        self.args = self.get_parser().parse_args()
        self._try_downloading_necessities(self.args.model_name)
        self.detection_model = self._build_detection_model()

        os.makedirs(self.args.output_folder, exist_ok=True)

    def _try_downloading_necessities(self, model_name):
        if self.args.model_file is None and model_name is not None:
            model_url = self.MODEL_URL[model_name]
            config_url = self.CONFIG_URL[model_name]
            self.args.model_file = model_url.split("/")[-1]
            self.args.config_file = config_url.split("/")[-1]
            if os.path.exists(self.args.model_file) and os.path.exists(
                self.args.config_file
            ):
                print(f"model and config file exists in directory: {os.getcwd()}")
                return
            print("Downloading model and configuration")
            download(model_url, ".", self.args.model_file)
            download(config_url, ".", self.args.config_file)

    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model_name", default="X-152", type=str, help="Model to use for detection"
        )
        parser.add_argument(
            "--model_file",
            default=None,
            type=str,
            help="Detectron model file. This overrides the model_name param.",
        )
        parser.add_argument(
            "--config_file", default=None, type=str, help="Detectron config file"
        )
        parser.add_argument(
            "--start_index", default=0, type=int, help="Index to start from "
        )
        parser.add_argument("--end_index", default=None, type=int, help="")
        parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
        parser.add_argument(
            "--num_features",
            type=int,
            default=100,
            help="Number of features to extract.",
        )
        parser.add_argument(
            "--output_folder", type=str, default="./output", help="Output folder"
        )
        parser.add_argument("--image_dir", type=str, help="Image directory or file")
        parser.add_argument(
            "--feature_name",
            type=str,
            help="The name of the feature to extract",
            default="fc6",
        )
        parser.add_argument(
            "--exclude_list",
            type=str,
            help="List of images to be excluded from feature conversion. "
            + "Each image on a new line",
            default="./list",
        )
        parser.add_argument(
            "--confidence_threshold",
            type=float,
            default=0,
            help="Threshold of detection confidence above which boxes will be selected",
        )
        parser.add_argument(
            "--background",
            action="store_true",
            help="The model will output predictions for the background class when set",
        )
        return parser

    def _build_detection_model(self):
        cfg.merge_from_file(self.args.config_file)
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load(self.args.model_file, map_location=torch.device("cpu"))

        load_state_dict(model, checkpoint.pop("model"))

        model.to("cuda")
        model.eval()
        return model

    def _image_transform(self, path):
        # img = Image.open(path)
        # img = cv2.imread(path)
        im_b64 = path
        im_bytes = base64.b64decode(im_b64)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        
        im = np.array(img).astype(np.float32)

        if im.shape[-1] > 3:
            im = np.array(img.convert("RGB")).astype(np.float32)

        # IndexError: too many indices for array, grayscale images
        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)

        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_height = im_shape[0]
        im_width = im_shape[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        # Scale based on minimum size
        im_scale = self.MIN_SIZE / im_size_min

        # Prevent the biggest axis from being more than max_size
        # If bigger, scale it down
        if np.round(im_scale * im_size_max) > self.MAX_SIZE:
            im_scale = self.MAX_SIZE / im_size_max

        im = cv2.resize(
            im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)

        im_info = {"width": im_width, "height": im_height}

        return img, im_scale, im_info

    def _process_feature_extraction(
        self, output, im_scales, im_infos, feature_name="fc6", conf_thresh=0
    ):
        batch_size = len(output[0]["proposals"])
        n_boxes_per_image = [len(boxes) for boxes in output[0]["proposals"]]
        score_list = output[0]["scores"].split(n_boxes_per_image)
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        feats = output[0][feature_name].split(n_boxes_per_image)
        cur_device = score_list[0].device

        feat_list = []
        info_list = []

        for i in range(batch_size):
            dets = output[0]["proposals"][i].bbox / im_scales[i]
            scores = score_list[i]
            max_conf = torch.zeros(scores.shape[0]).to(cur_device)
            conf_thresh_tensor = torch.full_like(max_conf, conf_thresh)
            start_index = 1
            # Column 0 of the scores matrix is for the background class
            if self.args.background:
                start_index = 0
            for cls_ind in range(start_index, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.5)
                max_conf[keep] = torch.where(
                    # Better than max one till now and minimally greater
                    # than conf_thresh
                    (cls_scores[keep] > max_conf[keep])
                    & (cls_scores[keep] > conf_thresh_tensor[keep]),
                    cls_scores[keep],
                    max_conf[keep],
                )

            sorted_scores, sorted_indices = torch.sort(max_conf, descending=True)
            num_boxes = (sorted_scores[: self.args.num_features] != 0).sum()
            keep_boxes = sorted_indices[: self.args.num_features]
            feat_list.append(feats[i][keep_boxes])
            bbox = output[0]["proposals"][i][keep_boxes].bbox / im_scales[i]
            # Predict the class label using the scores
            objects = torch.argmax(scores[keep_boxes][:, start_index:], dim=1)

            info_list.append(
                {
                    "bbox": bbox.cpu().numpy(),
                    "num_boxes": num_boxes.item(),
                    "objects": objects.cpu().numpy(),
                    "cls_prob": scores[keep_boxes][:, start_index:].cpu().numpy(),
                    "image_width": im_infos[i]["width"],
                    "image_height": im_infos[i]["height"],
                }
            )

        return feat_list, info_list

    def get_detectron_features(self, image_paths, begin_idx):
        img_tensor, im_scales, im_infos = [], [], []

        errors = set()
        for ix, image_path in enumerate(image_paths):
            try:
                im, im_scale, im_info = self._image_transform(image_path)
                img_tensor.append(im)
                im_scales.append(im_scale)
                im_infos.append(im_info)
            except:
                errors.add(ix) 
        # Image dimensions should be divisible by 32, to allow convolutions
        # in detector to work
        if len(img_tensor) == 0:
            return ([], []), errors
#        import pickle
#        with open('img_tensor.pkl', 'wb') as f:
#            pickle.dump(img_tensor, f)
#        with open('errors.txt', 'w') as f:
#            for e in errors:
#                f.write(str(begin_idx+e) + '\n')
#        with open('oks.txt', 'w') as f:
#            for e in [ix for ix in range(len(image_paths)) if ix not in errors]:
#                f.write(str(begin_idx+e) + '\n')
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to("cuda")

        with torch.no_grad():
            output = self.detection_model(current_img_list)

        feat_list = self._process_feature_extraction(
            output,
            im_scales,
            im_infos,
            self.args.feature_name,
            self.args.confidence_threshold,
        )

        return feat_list, errors

    def _save_feature(self, file_name, feature, info):
        file_base_name = os.path.basename(file_name)
        file_base_name = file_base_name.split(".")[0]
        info_file_base_name = file_base_name + "_info.npy"
        file_base_name = file_base_name + ".npy"

        np.save(
            os.path.join(self.args.output_folder, file_base_name), feature.cpu().numpy()
        )
        np.save(os.path.join(self.args.output_folder, info_file_base_name), info)

    def extract_features(self):
        image_dir = self.args.image_dir
        df = pd.read_csv(os.path.join(image_dir, 'chosen_langs.tsv'), sep='\t', usecols=['base64_image'])
        images = list(df['base64_image'].values)
        
#        out_dir = os.path.join(image_dir, 'wit_%d_resnext101' % args.chunk)
#        os.makedirs(out_dir, exist_ok=True)

        start_idx = 0
#        files = [os.path.join(out_dir, 'wit_%d' % (self.start_idx + idx) for idx in range(len(images))]
        files = ['wit-test_%d' % (start_idx + idx) for idx in range(len(images))]

        finished = 0
        total = len(files)

        for chunk, begin_idx in chunks(images, self.args.batch_size):
            (features, infos), errs = self.get_detectron_features(chunk, begin_idx)
            ok_idx = 0
            for idx in range(len(chunk)):
                if idx not in errs:
                    self._save_feature(files[begin_idx+idx], features[ok_idx], infos[ok_idx])
                    ok_idx += 1
            finished += len(chunk)

            if finished % 200 == 0:
                print(f"Processed {finished}/{total}")


if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    feature_extractor.extract_features()
