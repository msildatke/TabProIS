""" class Utils """
import json
import os.path
from json import JSONDecodeError
from typing import List

from errors import ResultFileDoesNotExistError, ResultFileNotParsableError
from models.coco_annotation import COCOAnnotation
from models.coco_bbox import COCOBbox


class Utils:
  """ Parses results as COCO annotations """

  @staticmethod
  def parse_result_file(file_path: str) -> List[COCOAnnotation]:
    """ Parses a text file containing results as COCO annotations.

    Args:
      file_path: Path to text file.

    Returns:
      List of COCO Annotations.

    Raises:
      ResultFileDoesNotExistError
      ResultFileNotParsableError
    """
    if not os.path.isfile(file_path):
      raise ResultFileDoesNotExistError("Result file does not exist.")

    with open(file_path, "r", encoding="utf-8") as f:
      try:
        annotations = json.loads(f.read())

        if isinstance(annotations, dict):
          annotations = annotations["annotations"]

        return [
          COCOAnnotation(
            image_id=a["image_id"],
            category_id=a["category_id"],
            bbox=COCOBbox(
              top_left_x=a["bbox"][0],
              top_left_y=a["bbox"][1],
              width=a["bbox"][2],
              height=a["bbox"][3]
            ),
            score=a["score"]
          )
          for a in annotations
        ]
      except (JSONDecodeError, KeyError) as e:
        raise ResultFileNotParsableError(f"Can not parse file: {str(e)}") from e

