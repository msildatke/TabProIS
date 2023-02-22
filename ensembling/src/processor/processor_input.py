""" class ProcessorInput """


class ProcessorInput:
  file_path: str
  min_confidence: float

  def __init__(self, file_path: str, min_confidence: float):
    self.file_path = file_path
    self.min_confidence = min_confidence
