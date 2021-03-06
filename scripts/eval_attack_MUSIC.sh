#!/bin/bash

OPTS=""
OPTS+="--mode eval "
OPTS+="--id MUSIC_av_reg-resnet18-anet-concat-learn-frames1stride8-maxpool-epoch100-step30_60_90 "
OPTS+="--list_train data/train.csv "
OPTS+="--list_val data/test.csv "

# Models
OPTS+="--arch_frame resnet18 "
OPTS+="--arch_sound anet "
OPTS+="--arch_classifier concat "
OPTS+="--img_pool maxpool "
OPTS+="--dic learn "

# attack type
OPTS+="--attack_type fsgm "

# logscale in frequency
OPTS+="--num_mix 1 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--num_frames 1 "
OPTS+="--stride_frames 8 "
OPTS+="--frameRate 8 "

# audio-related
OPTS+="--audLen 65535 " # 65535
OPTS+="--audRate 11025 " #11025

OPTS+="--num_gpus 3 "
OPTS+="--workers 32 "
OPTS+="--batch_size_per_gpu 12 "

python -u main_attack.py $OPTS