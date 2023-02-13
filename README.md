# CAD Transformer

## Source

```bash
https://github.com/VITA-Group/CADTransformer
```

## Install

```bash
conda create -n cadt python=3.7
conda activate cadt
./setup.sh
```

## Download

```bash
https://github.com/HRNet/HRNet-Image-Classification -> HRNet-W48-C
->
/home/chli/chLi/HRNet/hrnetv2_w48_imagenet_pretrained.pth
```

```bash
https://floorplancad.github.io/ -> Train set1, Train set2, Test set
->
/home/chli/chLi/FloorPlanCAD/
  |
  |-train-00/
  |  |-coco_vis/
  |  |-*.svg
  |  |-*.png
  |
  |-train-01/
  |  |-coco_vis/
  |  |-*.svg
  |  |-*.png
  |
  |-test-00/
  |  |-coco_vis/
  |  |-*.svg
  |  |-*.png
```

## Prepare

```bash
./pre.sh
```

## Train

```bash
./train.sh
```

## Eval

```bash
./eval.sh
```

## Run

```bash
python demo.py
```

## Enjoy it~

