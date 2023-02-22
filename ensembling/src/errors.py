""" Errors """


class NoValidBboxError(Exception):
  """ Throwable error if Bbox is not valid """


class NoIntersectionError(Exception):
  """ Throwable error if there is no intersection between two bounding boxes """


class NoValidAnnotationError(Exception):
  """ Throwable error if annotation is not valid """


class NotTheSameImageError(Exception):
  """ Throwable error if ensemble of two annotations with different images """


class NotTheSameCategoryError(Exception):
  """ Throwable error if ensemble of two annotations with different categories """


class ResultFileDoesNotExistError(Exception):
  """ Throwable error if result file does not exist """


class ResultFileNotParsableError(Exception):
  """ Throwable error if result file is not parsable """
