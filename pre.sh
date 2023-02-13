# Demo data, not used
# python cad_transformer/Pre/download_data.py \
  # --data_save_dir /home/chli/chLi/CADTransformer

python cad_transformer/Pre/svg2png.py \
  --data_save_dir /home/chli/chLi/FloorPlanCAD \
  --scale 7 \
  --cvt_color

python cad_transformer/Pre/preprocess_svg.py \
  -i /home/chli/chLi/FloorPlanCAD/svg/train \
  -o /home/chli/chLi/FloorPlanCAD/npy/train

python cad_transformer/Pre/preprocess_svg.py \
  -i /home/chli/chLi/FloorPlanCAD/svg/test \
  -o /home/chli/chLi/FloorPlanCAD/npy/test

python cad_transformer/Pre/preprocess_svg.py \
  -i /home/chli/chLi/FloorPlanCAD/svg/val \
  -o /home/chli/chLi/FloorPlanCAD/npy/val

