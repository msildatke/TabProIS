""" main module """
from typing import Tuple, List

import click

from processor.ensemble_processor import EnsembleProcessor


@click.command()
@click.option("--model_result", nargs=2, type=(str, float), multiple=True,
              help="Path to file containing results of first model.")
@click.option("--target", help="Path to target file containing ensemble results.")
@click.option("--threshold", default=0.5, help="IoU-threshold two detections of different models are declared as one.")
@click.option("--strategy", default=1, help="Ensemble strategy: 1=Affirmative, 2=Consensus, 3=Unanimous")
@click.option("--method", default=1, help="1=NMS, 2=WBF")
def main(
    model_result: List[Tuple[str, float]],
    target: str,
    threshold: float,
    strategy: int,
    method: int
):
  """ Triggers the EnsembleProcessor with console arguments.

  Args:
    model_result: Path to file containing results of first model.
    target: Path to target file containing ensemble results.
    threshold: IoU-threshold two detections of different models are declared as one.
    strategy: Ensemble strategy: 1=Affirmative, 2=Consensus, 3=Unanimous
    method: 1=NMS, 2=WBF

  Returns:
    None

  """
  ensemble_processor = EnsembleProcessor()

  for m in model_result:
    ensemble_processor.add_input(
      file_path=m[0],
      min_confidence=m[1]
    )

  annotations = ensemble_processor.process(
    result_file_path=target, iou_threshold=threshold, strategy=strategy, method=method
  )
  click.echo(f"Target file generated with {len(annotations)} ensemble results.")


if __name__ == "__main__":
  main()  # pylint: disable-all
