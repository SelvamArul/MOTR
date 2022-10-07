shopt -s expand_aliases
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

# for MOT17
alias xpython="singularity run -B /home/nfs/inf6/data/models,/home/nfs/inf6/data/datasets/YCB_VIDEO_DATASET,/home/nfs/inf6/data/datasets/ycbv_coco,$HOME/.local/bin --env PATH=$HOME/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin --nv /home/nfs/inf6/data/singularity_images/motr_image.sif python"
EXP_DIR=exps/vanilla_detr_pose
xpython main.py \
    --meta_arch motr \
    --dataset_file ycbv\
    --dataset_path '/home/nfs/inf6/data/datasets/YCB_VIDEO_DATASET/YCB_Video_Dataset/data/'\
    --dataset_desc_file_train '/home/user/periyasa/workspace/MOTR/datasets/ycbv_train_desc_train.txt'\
    --dataset_desc_file_val '/home/user/periyasa/workspace/MOTR/datasets/ycbv_train_desc_val.txt'\
    --pretrained '/home/user/periyasa/workspace/MOTR/exps/vanilla_detr_pose_31-Aug-2022_09:40/checkpoint.pth'\
    --resume '/home/user/periyasa/workspace/MOTR/exps/vanilla_detr_pose_31-Aug-2022_09:40/checkpoint.pth'\
    --epoch 100 \
    --num_workers 4 \
    --lr_drop 100 \
    --lr 3e-4 \
    --lr_backbone 2e-5 \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'fixed_interval' \
    --sample_interval 2 \
    --sampler_steps 2 4 6 \
    --sampler_lengths 4 5 5 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIM' \
    --extra_track_attn \
    --data_txt_path_train ./datasets/data_path/mot17.train \
    --data_txt_path_val ./datasets/data_path/mot17.train \
    --mot_path '/home/nfs/inf6/data/datasets/YCB_VIDEO_DATASET/' \
    --enable_pose \
    --sym_classes 13 16 19 20 21
