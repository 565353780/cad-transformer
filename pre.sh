DATASET_ROOT_PATH=/home/chli/chLi/FloorPlanCAD

# Demo data, not used
# python cad_transformer/Pre/download_data.py \
  # --data_save_dir /home/chli/chLi/CADTransformer

python cad_transformer/Pre/svg2png.py \
  --data_save_dir $DATASET_ROOT_PATH \
  --scale 7

python cad_transformer/Pre/preprocess_svg.py \
  -i $DATASET_ROOT_PATH/svg/train \
  -o $DATASET_ROOT_PATH/npy/train

python cad_transformer/Pre/preprocess_svg.py \
  -i $DATASET_ROOT_PATH/svg/test \
  -o $DATASET_ROOT_PATH/npy/test

python cad_transformer/Pre/preprocess_svg.py \
  -i $DATASET_ROOT_PATH/svg/val \
  -o $DATASET_ROOT_PATH/npy/val

