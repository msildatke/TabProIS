""" class COCOBbox """
from errors import NoValidBboxError
from models.corner_point_based_bbox import CornerPointBasedBbox


class COCOBbox:
  """ Represents a COCO bounding box. """

  top_left_x: float
  top_left_y: float
  width: float
  height: float

  def __init__(
      self,
      top_left_x: float,
      top_left_y: float,
      width: float,
      height: float
  ):
    """ Creates a new BBox instance.

    Args:
      top_left_x: x value of the top left corner.
      top_left_y: y value of the top left corner.
      width: Width of the Bbox.
      height: Height of the Bbox.

    Raises:
        NoValidBboxError

    """
    if top_left_x < 0 or top_left_y < 0 or width <= 0 or height <= 0:
      raise NoValidBboxError("No valid COCOBbox.")

    self.top_left_x = top_left_x
    self.top_left_y = top_left_y
    self.width = width
    self.height = height

  @staticmethod
  def from_corner_point_based_bbox(bbox: CornerPointBasedBbox):
    """ Creates a COCOBbox from a CoordinateBasedBbox instance. """
    return COCOBbox(
      top_left_x=bbox.top_left_x,
      top_left_y=bbox.top_left_y,
      width=bbox.bottom_right_x - bbox.top_left_x,
      height=bbox.bottom_right_y - bbox.top_left_y
    )

  def to_corner_point_based_bbox(self) -> CornerPointBasedBbox:
    """ Converts COCOBbox to CoordinateBasedBbox  """
    return CornerPointBasedBbox(
      top_left_x=self.top_left_x,
      top_left_y=self.top_left_y,
      bottom_right_x=self.top_left_x + self.width,
      bottom_right_y=self.top_left_y + self.height
    )

  def __eq__(self, other):
    """ Checks equality of two Bboxes """

    if not isinstance(other, COCOBbox):
      return False
    return self.top_left_x == other.top_left_x \
           and self.top_left_y == other.top_left_y \
           and self.width == other.width \
           and self.height == other.height
