shopt -s expand_aliases
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


#    --resume '/home/user/periyasa/workspace/MOTR/exps/ddp_ycbv_from_scratch_21-Mar-2022_12:41/checkpoint.pth' \
# for MOT17


alias xpython="singularity run -B /home/nfs/inf6/data/datasets/YCB_VIDEO_DATASET,/home/nfs/inf6/data/datasets/ycbv_coco --nv /home/cache/containers/motr_image.sif python"
EXP_DIR=exps/ycbv_ds_debug
xpython datasets/ycbv.py
