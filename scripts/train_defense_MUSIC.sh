#!/bin/bash

OPTS=""
OPTS+="--id MUSIC_av_reg_defense "
OPTS+="--list_train data/train.csv "
OPTS+="--list_val data/val.csv "

# Models
OPTS+="--arch_frame resnet18 "
OPTS+="--arch_sound anet "
OPTS+="--arch_classifier concat "
OPTS+="--img_pool maxpool "
OPTS+="--dic fix "

OPTS+="--weights_Dv data/ckpt/MUSIC_av_reg-resnet18-anet-concat-learn-frames1stride8-maxpool-epoch100-step30_60_90 "
OPTS+="--weights_Da data/ckpt/MUSIC_av_reg-resnet18-anet-concat-learn-frames1stride8-maxpool-epoch100-step30_60_90 "

# weights
OPTS+="--weights_sound data/ckpt/MUSIC_av_reg-resnet18-anet-concat-learn-frames1stride8-maxpool-epoch100-step30_60_90/sound_latest.pth "
OPTS+="--weights_frame data/ckpt/MUSIC_av_reg-resnet18-anet-concat-learn-frames1stride8-maxpool-epoch100-step30_60_90/frame_latest.pth "
OPTS+="--weights_classifier data/ckpt/MUSIC_av_reg-resnet18-anet-concat-learn-frames1stride8-maxpool-epoch100-step30_60_90/classifier_latest.pth "

# binary mask, BCE loss, weighted loss
OPTS+="--loss ce "
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

# learning params
OPTS+="--num_gpus 3 "
OPTS+="--workers 32 "
OPTS+="--batch_size_per_gpu 12 "
OPTS+="--lr_frame 1e-4 " #1e-4
OPTS+="--lr_sound 1e-4 " #1e-3
OPTS+="--lr_classifier 1e-4 " #1e-3
OPTS+="--num_epoch 100 "
OPTS+="--lr_steps 30 60 90 "

# display, viz
OPTS+="--disp_iter 5 "

python -u main_defense.py $OPTS
