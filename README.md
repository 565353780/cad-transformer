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

modify the pre.sh and run

```bash
./pre.sh
```

## Train

modify the cad_transformer/Config/args.py and run

```bash
python train.py
```

## Run

```bash
python demo.py
```

## Log

```bash
tensorboard --logdir ./logs --host 0.0.0.0
```

## GPU Memory

| max_prim | GPU Memory |
| :-: | :-: |
| 1000 | 5795M |
| 2000 | 6611M |
| 3000 | 7297M |
| 4000 | 8665M |

## Enjoy it~

