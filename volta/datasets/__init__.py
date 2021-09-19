# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .concept_cap_dataset import ConceptCapLoaderTrain, ConceptCapLoaderVal
from .vqa_dataset import VQAClassificationDataset
from .nlvr2_dataset import NLVR2Dataset
from .refer_expression_dataset import ReferExpressionDataset
from .retrieval_dataset import RetrievalDataset, RetrievalDatasetVal
from .vcr_dataset import VCRDataset
from .visual_entailment_dataset import VisualEntailmentDataset
from .refer_dense_caption import ReferDenseCpationDataset
from .visual_genome_dataset import GenomeQAClassificationDataset
from .gqa_dataset import GQAClassificationDataset
from .guesswhat_dataset import GuessWhatDataset
from .visual7w_pointing_dataset import Visual7wPointingDataset
from .guesswhat_pointing_dataset import GuessWhatPointingDataset
from .flickr_grounding_dataset import FlickrGroundingDataset
from .flickr30ke_ablation_dataset import FlickrVis4LangDataset, FlickrLang4VisDataset


__all__ = [
    "VQAClassificationDataset",
    "GenomeQAClassificationDataset",
    "ConceptCapLoaderTrain",
    "ConceptCapLoaderVal",
    "NLVR2Dataset",
    "ReferExpressionDataset",
    "RetrievalDataset",
    "RetrievalDatasetVal",
    "VCRDataset",
    "VisualEntailmentDataset",
    "GQAClassificationDataset",
    "GuessWhatDataset",
    "Visual7wPointingDataset",
    "GuessWhatPointingDataset",
    "FlickrGroundingDataset",
    "FlickrVis4LangDataset",
    "FlickrLang4VisDataset",
    "",
]

DatasetMapTrain = {
    "VQA": VQAClassificationDataset,
    "GenomeQA": GenomeQAClassificationDataset,
    "VCR_Q-A": VCRDataset,
    "VCR_QA-R": VCRDataset,
    "RetrievalCOCO": RetrievalDataset,
    "RetrievalFlickr30k": RetrievalDataset,
    "refcoco": ReferExpressionDataset,
    "refcoco+": ReferExpressionDataset,
    "refcocog": ReferExpressionDataset,
    "NLVR2": NLVR2Dataset,
    "VisualEntailment": VisualEntailmentDataset,
    "GQA": GQAClassificationDataset,
    "GuessWhat": GuessWhatDataset,
    "Visual7w": Visual7wPointingDataset,
    "GuessWhatPointing": GuessWhatPointingDataset,
    "FlickrGrounding": FlickrGroundingDataset,
}

DatasetMapEval = {
    "VQA": VQAClassificationDataset,
    "GenomeQA": GenomeQAClassificationDataset,
    "VCR_Q-A": VCRDataset,
    "VCR_QA-R": VCRDataset,
    "RetrievalCOCO": RetrievalDatasetVal,
    "RetrievalFlickr30k": RetrievalDatasetVal,
    "refcoco": ReferExpressionDataset,
    "refcoco+": ReferExpressionDataset,
    "refcocog": ReferExpressionDataset,
    "NLVR2": NLVR2Dataset,
    "VisualEntailment": VisualEntailmentDataset,
    "GQA": GQAClassificationDataset,
    "GuessWhat": GuessWhatDataset,
    "Visual7w": Visual7wPointingDataset,
    "GuessWhatPointing": GuessWhatPointingDataset,
    "FlickrGrounding": FlickrGroundingDataset,
}
