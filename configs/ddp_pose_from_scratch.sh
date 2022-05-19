shopt -s expand_aliases
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


#    --resume '/home/user/periyasa/workspace/MOTR/exps/ddp_ycbv_from_scratch_21-Mar-2022_12:41/checkpoint.pth' \
# for MOT17

# read num_gpus
dev=${CUDA_VISIBLE_DEVICES}
gpus=`echo $dev | tr -d ,`
if test ${#gpus} -lt 2
then
    echo "Error: Number of available gpus ${dev} is not enough for distributed training"
    exit 1
fi


alias xpython="singularity run -B /home/nfs/inf6/data/datasets/YCB_VIDEO_DATASET,/home/nfs/inf6/data/datasets/ycbv_coco --nv /home/cache/containers/motr_image.sif python"
EXP_DIR=exps/ddp_pose_from_scratch
xpython -m torch.distributed.launch --nproc_per_node=${#gpus} \
    --use_env main.py \
    --meta_arch motr \
    --dataset_file ycbv\
    --dataset_path '/home/nfs/inf6/data/datasets/YCB_VIDEO_DATASET/YCB_Video_Dataset/data/'\
    --dataset_desc_file_train '/home/user/periyasa/workspace/MOTR/datasets/ycbv_train_desc_train.txt'\
    --dataset_desc_file_val '/home/user/periyasa/workspace/MOTR/datasets/ycbv_train_desc_val.txt'\
    --resume '/home/user/periyasa/workspace/MOTR/exps/ddp_pose_from_scratch_24-Apr-2022_08:58/checkpoint.pth'\
    --epoch 100 \
    --num_workers 4 \
    --with_box_refine \
    --lr_drop 100 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 2 \
    --sampler_steps 2 5 10 \
    --sampler_lengths 4 5 6 6 \
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
