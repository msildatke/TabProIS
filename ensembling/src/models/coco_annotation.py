""" class COCOAnnotation """
from errors import NoValidAnnotationError
from models.coco_bbox import COCOBbox
from models.corner_point_based_annotation import CornerPointBasedAnnotation


class COCOAnnotation:
  """ Represents a COCO annotation """

  image_id: int
  category_id: int
  bbox: COCOBbox
  score: float

  def __init__(self, image_id: int, category_id: int, bbox: COCOBbox, score: float):
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

  def to_corner_point_based_annotation(self) -> CornerPointBasedAnnotation:
    """ Convert annotation to CornerPointBasedAnnotation """
    return CornerPointBasedAnnotation(
      image_id=self.image_id,
      category_id=self.category_id,
      bbox=self.bbox.to_corner_point_based_bbox(),
      score=self.score
    )

  def to_json(self):
    """ Convert Annotation to JSON """
    return dict(
      image_id=self.image_id,
      category_id=self.category_id,
      bbox=[self.bbox.top_left_x, self.bbox.top_left_y, self.bbox.width, self.bbox.height],
      score=self.score
    )

  def __eq__(self, other):
    """ Checks equality of two Annotations """

    if not isinstance(other, COCOAnnotation):
      return False
    return self.image_id == other.image_id \
           and self.category_id == other.category_id \
           and self.bbox == other.bbox \
           and self.score == other.score

  def __hash__(self):
    return hash(
      str(self.image_id) + \
      str(self.category_id) + \
      str(self.score) + \
      str(self.bbox.width) + \
      str(self.bbox.height) + \
      str(self.bbox.top_left_x) + \
      str(self.bbox.top_left_y)
    )
