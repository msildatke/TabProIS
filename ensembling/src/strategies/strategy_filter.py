""" class StrategyFilter """
import itertools
from typing import List, Tuple

from models.coco_annotation import COCOAnnotation


class StrategyFilter:
  """ Filters model results depending on the ensemble strategy. """

  @staticmethod
  def affirmative(model_results: List[List[COCOAnnotation]]) -> List[COCOAnnotation]:
    """ Filters results for affirmative strategy. All annotations are candidates. """
    filtered_annotations = []
    for model_result in model_results:
      filtered_annotations.extend(model_result)
    return filtered_annotations

  @staticmethod
  def _unanimous(model_results: List[List[COCOAnnotation]], iou_threshold: float = 0.5):
    """ Filters results for unanimous strategy.
    Only those annotations are candidates that are in every model result. """
    filtered_annotations = []

    for result_combination in (itertools.product(*model_results)):
      all_boxes_overlap_and_reach_iou_threshold = True
      for result_pair in itertools.combinations(result_combination, 2):
        result_pair: Tuple[COCOAnnotation, COCOAnnotation]
        if result_pair[0].bbox.to_corner_point_based_bbox().intersection_over_union(
            result_pair[1].bbox.to_corner_point_based_bbox()
        ) < iou_threshold:
          all_boxes_overlap_and_reach_iou_threshold = False
          break
      if all_boxes_overlap_and_reach_iou_threshold:
        for annotation in result_combination:
          if annotation not in filtered_annotations: filtered_annotations.append(annotation)

    return filtered_annotations

  @staticmethod
  def _consensus(model_results: List[List[COCOAnnotation]], iou_threshold: float = 0.5):
    """ Filters results for consensus strategy.
    Only those annotations are candidates that are in at least no_of_models /2  results. """
    filtered_annotations = []

    if len(model_results) == 2:
      return StrategyFilter.unanimous(model_results)

    for model_result in model_results:
      for main_annotation in model_result:
        number_of_overlaps_reaching_iou_threshold = 1
        second_level_model_results = model_results[:]
        second_level_model_results.remove(model_result)
        for second_level_annotations in list(itertools.product(*second_level_model_results)):
          for second_level_annotation in second_level_annotations:
            iou = main_annotation.bbox.to_corner_point_based_bbox().intersection_over_union(
              second_level_annotation.bbox.to_corner_point_based_bbox())
            if iou >= iou_threshold:
              number_of_overlaps_reaching_iou_threshold += 1
          if number_of_overlaps_reaching_iou_threshold >= len(model_results) / 2:
            if main_annotation not in filtered_annotations: filtered_annotations.append(main_annotation)

    return filtered_annotations

  @staticmethod
  def unanimous(model_results: List[List[COCOAnnotation]], iou_threshold: float = 0.5):
    """ Filters results for unanimous strategy.
    Only those annotations are candidates that are in every model result. """
    filtered_annotations = []

    image_ids = []

    for model_result in model_results:
      for annotation in model_result:
        image_ids.append(annotation.image_id)

    for image_id in image_ids:
      model_annotations = []
      for model_result in model_results:
        model_annotation = []
        for annotation in model_result:
          if annotation.image_id == image_id: model_annotation.append(annotation)
        model_annotations.append(model_annotation)
      filtered_annotations.extend(StrategyFilter._unanimous(model_annotations, iou_threshold))

    return filtered_annotations

  @staticmethod
  def consensus(model_results: List[List[COCOAnnotation]], iou_threshold: float = 0.5):
    """ Filters results for unanimous strategy.
    Only those annotations are candidates that are in every model result. """
    filtered_annotations = []

    image_ids = []

    for model_result in model_results:
      for annotation in model_result:
        image_ids.append(annotation.image_id)

    for image_id in set(image_ids):
      model_annotations = []
      for model_result in model_results:
        model_annotation = []
        for annotation in model_result:
          if annotation.image_id == image_id: model_annotation.append(annotation)
        model_annotations.append(model_annotation)
      filtered_annotations.extend(StrategyFilter._consensus(model_annotations, iou_threshold))

    return filtered_annotations
