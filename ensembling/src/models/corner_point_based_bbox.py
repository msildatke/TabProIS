""" Corner-Point based Bbox"""
from errors import NoValidBboxError, NoIntersectionError


class CornerPointBasedBbox:
  """ Represents a bounding box with top_left and bottom_right corner """

  top_left_x: float
  top_left_y: float
  bottom_right_x: float
  bottom_right_y: float

  def __init__(self, top_left_x: float, top_left_y: float, bottom_right_x: float, bottom_right_y: float):
    """ Creates a new CoordinateBasedBbox instance. """

    if bottom_right_y <= top_left_y or top_left_x >= bottom_right_x \
        or top_left_x < 0 or top_left_y < 0 or bottom_right_x < 0 or bottom_right_y < 0:
      raise NoValidBboxError("coordinates must be >= 0")

    self.top_left_x = top_left_x
    self.top_left_y = top_left_y
    self.bottom_right_x = bottom_right_x
    self.bottom_right_y = bottom_right_y

  def area(self) -> float:
    """ Calculates the area of a bbox """
    return (self.bottom_right_x - self.top_left_x) * (self.bottom_right_y - self.top_left_y)

  def intersection(self, other: "CornerPointBasedBbox") -> "CornerPointBasedBbox":
    """ Calculates the intersection of two bboxes.

    Args:
      other: Other bounding box.

    Returns:
      Bbox object representing intersection.

    Raises:
      NoIntersectionError

    """
    try:
      top_left_x = max(self.top_left_x, other.top_left_x)
      top_left_y = max(self.top_left_y, other.top_left_y)
      bottom_right_x = min(self.bottom_right_x, other.bottom_right_x)
      bottom_right_y = min(self.bottom_right_y, other.bottom_right_y)

      intersection = CornerPointBasedBbox(
        top_left_x=top_left_x,
        top_left_y=top_left_y,
        bottom_right_x=bottom_right_x,
        bottom_right_y=bottom_right_y
      )

      return intersection
    except NoValidBboxError as e:
      raise NoIntersectionError("Bboxes do not intersect.") from e

  def intersection_over_union(self, other: "CornerPointBasedBbox") -> float:
    """ Calculates IoU with another Bbox.

    Args:
      other: Bbox IoU is calculated for.

    Returns:
      IoU between 0 and 1.

    """
    try:
      intersection = self.intersection(other)
      return intersection.area() / (self.area() + other.area() - intersection.area())
    except NoIntersectionError:
      return 0

  def __eq__(self, other):
    """ Checks if two Bboxes are equal """
    if not isinstance(other, CornerPointBasedBbox):
      return False
    return self.top_left_x == other.top_left_x \
           and self.top_left_y == other.top_left_y \
           and self.bottom_right_x == other.bottom_right_x \
           and self.bottom_right_y == other.bottom_right_y
