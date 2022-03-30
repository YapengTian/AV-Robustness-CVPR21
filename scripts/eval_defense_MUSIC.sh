#!/bin/bash

OPTS=""
OPTS+="--mode eval "
OPTS+="--id MUSIC_av_reg_defense-resnet18-anet-concat-fix-frames1stride8-maxpool-epoch100-step30_60_90 "
OPTS+="--list_train data/train.csv "
OPTS+="--list_val data/test.csv "

# Models
OPTS+="--arch_frame resnet18 "
OPTS+="--arch_sound anet "
OPTS+="--arch_classifier concat "
OPTS+="--img_pool maxpool "

OPTS+="--weights_Dv data/ckpt/MUSIC_av_reg-resnet18-anet-concat-learn-frames1stride8-maxpool-epoch100-step30_60_90 "
OPTS+="--weights_Da data/ckpt/MUSIC_av_reg-resnet18-anet-concat-learn-frames1stride8-maxpool-epoch100-step30_60_90 "

# attack type
OPTS+="--attack_type mim "

# logscale in frequency
OPTS+="--num_mix 1 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--num_gpus 4 "
OPTS+="--batch_size_per_gpu 12 "
OPTS+="--num_frames 1 "
OPTS+="--stride_frames 8 "
OPTS+="--frameRate 8 "

# Sparse solver
OPTS+="--alpha_a  1e-6 "
OPTS+="--alpha_v  6e-6 "

# audio-related
OPTS+="--audLen 65535 " # 65535
OPTS+="--audRate 11025 " #11025


python -u main_defense.py $OPTS