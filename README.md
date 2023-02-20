# TabProIS: A Transfer Learning-Based Model for Detecting Tables in Product Information Sheets

This repository contains the trained model files in PyTorch (.pth) format which can be used to reproduce the findings in our paper. If you want to cite our work, please refer to the section below.

## Guides for TableBank (detectron2), CDeCNet (mmdetection) and YOLOv5

**TableBank (detectron2)**

[project page](https://github.com/doc-analysis/TableBank)

detectron2 documentation
- [overview](https://detectron2.readthedocs.io/en/latest/index.html)
- [requirements and installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
- [inference with existing dataset](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html)

**CDeCNet (mmdetection)**

[project page](https://github.com/mdv3101/CDeCNet)

mmdetection documentation
- [overview](https://github.com/open-mmlab/mmdetection)
- [requirements and installation](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md)
- [inference and training with existing dataset](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md)

**YOLOv5**
- [overview](https://github.com/ultralytics/yolov5)
- [requirements, installation and inference](https://github.com/ultralytics/yolov5#documentation)

## Requirements and Setup

Our models were trained with three architectures, which are TableBank, CDeCNet and YOLOv5. Please refer to the respective guides for their individual requirements. We recommend to create three different virtual environments (preferably with conda), one for each.

For setting up the environments as well as a list of all requirements, we encourage you to follow the official guides.

The ensembling code has little requirements which can be found in the file [`requirements.txt`](ensembling/requirements.txt).

## Reproducing our Results

### Training and Prediction

In order to reproduce our results, you need to obtain our pretrained models. Then, you can run predictions with our dataset according to the official guides.

YOLO requires the dataset to be in a format different from COCO. We provide a tool to convert the dataset to a YOLO compliant format.

### Ensembling

The ensembling program can be called as follows:

```
PYTHONPATH=src python main.py \
    --results_1 path --results_2 path --target output_path \
    --results_1_threshold float --results_2_threshold float \
    [--threshold 0.5 [--strategy 0]]
```

Options:

- **--results_1** Path to file containing results of first model
- **--results_2** Path to file containing results of second model
- **--target** Path to target output file
- **--results_1_threshold** Confidence threshold for results of first model
- **--results_2_threshold** Confidence threshold for results of second model
- **--threshold** _[optional, default: 0.5]_ IoU-threshold above which two detections of different models are treated as the same
- **--strategy** _[optional, default: 0]_ Ensembling strategy: 0 = Affirmative, 1 = Unanimous

### COCO 2 YOLO

TODO: add script and guide

## Models

TODO: Hosting and links

## Cite This

```tex
@article {
    # ... coming soon
}
```