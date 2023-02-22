""" class CornerPointBasedAnnotation """
from errors import NoValidAnnotationError
from models.corner_point_based_bbox import CornerPointBasedBbox


class CornerPointBasedAnnotation:
  """ Represents a COCO annotation with CornerPointBasedBboxes """

  image_id: int
  category_id: int
  bbox: CornerPointBasedBbox
  score: float

  def __init__(self, image_id: int, category_id: int, bbox: CornerPointBasedBbox, score: float):
    """ Creates a new Annotation instance.

    Args:
      image_id: id of the annotated image.
      category_id: id of the labeled category (class).
      bbox: Predicted bounding box.
      score: Confidence of the prediction.

    Raises:
      NoValidAnnotationError
    """
    if image_id < 0 or category_id < 0 or score <= 0:
      raise NoValidAnnotationError("Not a valid Annotation.")

    self.image_id = image_id
    self.category_id = category_id
    self.bbox = bbox
    self.score = score

  def __eq__(self, other):
    """ Checks equality of two Annotations """

    if not isinstance(other, CornerPointBasedAnnotation):
      return False
    return self.image_id == other.image_id \
           and self.category_id == other.category_id \
           and self.bbox == other.bbox \
           and self.score == other.score
