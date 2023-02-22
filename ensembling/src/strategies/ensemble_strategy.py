""" class EnsembleDetector """
from typing import List

from ensemble_boxes import nms, weighted_boxes_fusion

from models.coco_annotation import COCOAnnotation
from models.coco_bbox import COCOBbox
from models.corner_point_based_bbox import CornerPointBasedBbox

WIDTH = 20000
HEIGHT = 20000


class EnsembleStrategy:
  """ Detects tables using ensembles of different models """

  @staticmethod
  def _do_ensemble(annotations: List[COCOAnnotation], iou_threshold: float = 0.5, method: int = 1):
    """ Do Ensemble """
    ensemble_annotations = []

    image_ids = []

    images = {a.image_id for a in annotations}

    for image in images:
      boxes = []
      scores = []
      labels = []

      image_annotations = [a for a in annotations if a.image_id == image]

      for annotation in image_annotations:
        image_ids.append(annotation.image_id)
        boxes.append(
          (
            annotation.bbox.to_corner_point_based_bbox().top_left_x / WIDTH,
            annotation.bbox.to_corner_point_based_bbox().top_left_y / HEIGHT,
            annotation.bbox.to_corner_point_based_bbox().bottom_right_x / WIDTH,
            annotation.bbox.to_corner_point_based_bbox().bottom_right_y / HEIGHT,
          )
        )

        scores.append(annotation.score)
        labels.append(annotation.category_id)

      if method == 1:
        boxes, scores, labels = nms([boxes], [scores], [labels], iou_thr=iou_threshold)
      else:
        boxes, scores, labels = weighted_boxes_fusion([boxes], [scores], [labels], iou_thr=iou_threshold)

      for i in range(len(boxes)):
        ensemble_annotations.append(
          COCOAnnotation(
            image_id=image,
            category_id=int(labels[i]),
            bbox=COCOBbox.from_corner_point_based_bbox(
              bbox=CornerPointBasedBbox(
                top_left_x=float(boxes[i][0] * WIDTH),
                top_left_y=float(boxes[i][1] * HEIGHT),
                bottom_right_x=float(boxes[i][2] * WIDTH),
                bottom_right_y=float(boxes[i][3] * HEIGHT)
              )
            ),
            score=float(scores[i])
          )
        )

    return ensemble_annotations

  @staticmethod
  def non_maximum_suppressor(annotations: List[COCOAnnotation], iou_threshold: float = 0.5) -> List[COCOAnnotation]:
    """ Calculates ensemble based on Non-Maximum Suppressor (NMS) method """
    return EnsembleStrategy._do_ensemble(annotations=annotations, iou_threshold=iou_threshold, method=1)

  @staticmethod
  def weighted_box_fusion(annotations: List[COCOAnnotation], iou_threshold: float = 0.5) -> List[COCOAnnotation]:
    """ Calculates ensemble based on Weighted Box Fusion (WBF) method """
    return EnsembleStrategy._do_ensemble(annotations=annotations, iou_threshold=iou_threshold, method=2)
