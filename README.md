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

### Ensembling

The ensembling program can be called as follows:

```shell
PYTHONPATH=src python main.py \
    --model_result path thresh path thresh [path thresh [..]]
    --target output_path [--threshold 0.5 [--strategy 0 [--method 1]]]
```

Options:

- **--model_result** Paths to files containing results of the models and respective confidence thresholds
- **--target** Path to target output file
- **--threshold** _[optional, default: 0.5]_ IoU-threshold above which two detections of different models are treated as the same
- **--strategy** _[optional, default: 1]_ Ensembling strategy: 1 = Affirmative, 2 = Consensus, 3 = Unanimous
- **--method** _[optional, default: 1]_ Ensembling method: 1 = NMS, 2 = WBF

In order to successfully run ensembling on the output of YOLO models, the output needs to be converted back to COCO first. You can, for example, use [Taeyoung96's YOLO to COCO format converter](https://github.com/Taeyoung96/Yolo-to-COCO-format-converter) for this task.

## Models

TODO: Hosting and links

## Cite This

```tex
@article {
    # ... coming soon
}
```