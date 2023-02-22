""" class EnsembleProcessor """
import json
from typing import List

from models.coco_annotation import COCOAnnotation
from processor.processor_input import ProcessorInput
from strategies.ensemble_strategy import EnsembleStrategy
from strategies.strategy_filter import StrategyFilter
from utils import Utils


class EnsembleProcessor:
  """ Processes two result lists of different models and processes the ensemble. """

  processor_inputs: List[ProcessorInput]

  def __init__(self):
    """ Creates a new EnsembleProcessor instance. """
    self.processor_inputs = []

  def add_input(self, file_path: str, min_confidence: float):
    """ Add input. """
    self.processor_inputs.append(
      ProcessorInput(file_path=file_path, min_confidence=min_confidence)
    )

  def process(
      self,
      result_file_path: str = None,
      strategy: int = 1,
      method: int = 1,
      iou_threshold: float = 0.5
  ) -> List[COCOAnnotation]:
    """ Performs the ensemble process.

    Args:
      result_file_path: output file path containing results
      strategy: 1=Affirmative, 2=Consensus, 3=Unanimous
      method: 1=Non-Maximum Suppressor, 2=Weighted Box Fusion
      iou_threshold: Threshold for boxes to be processed as one-

    Returns:
      List of COCO annotations that also are exported.

    """

    model_results = []

    for processor_input in self.processor_inputs:
      model_result = []
      for model_annotation in  Utils.parse_result_file(file_path=processor_input.file_path):
        if model_annotation.score >= processor_input.min_confidence: model_result.append(model_annotation)
      model_results.append(model_result)

    if strategy == 1:
      filtered_results = StrategyFilter.affirmative(model_results)
    elif strategy == 2:
      filtered_results = StrategyFilter.consensus(model_results)
    else:
      filtered_results = StrategyFilter.unanimous(model_results)

    if method == 1:
      ensemble_annotations = EnsembleStrategy.non_maximum_suppressor(
        annotations=filtered_results, iou_threshold=iou_threshold
      )
    else:
      ensemble_annotations = EnsembleStrategy.weighted_box_fusion(
        annotations=filtered_results, iou_threshold=iou_threshold
      )

    if result_file_path:
      with open(result_file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps([a.to_json() for a in ensemble_annotations], indent=4))

    return ensemble_annotations
