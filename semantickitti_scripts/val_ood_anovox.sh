name=cylinder_asym_networks
gpuid=1

CUDA_VISIBLE_DEVICES=${gpuid}  python -u val_cylinder_asym_ood_anovox.py \
--config '/home/lukasnroessler/Projects/Open_world_3D_semantic_segmentation/config/anovox_val.yaml'