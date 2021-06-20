#!/bin/bash

OPTS=""
OPTS+="--id Kinetics_av_reg_FB_orifeat "
OPTS+="--list_train data/train_kinetics.csv "
OPTS+="--list_val data/val_kinetics.csv "

# Models
OPTS+="--arch_frame resnet18 "
OPTS+="--arch_sound anet "
OPTS+="--arch_classifier visual "
OPTS+="--img_pool maxpool "
OPTS+="--dic fix "

OPTS+="--cls_num 27 "
OPTS+="--categories laughing playing_clarinet singing
       playing_harmonica playing_keyboard playing_xylophone playing_bass_guitar
       tapping_guitar playing_drums playing_piano ripping_paper playing_saxophone
       tickling playing_trumpet tapping_pen playing_organ tap_dancing playing_accordion
       blowing_nose shuffling_cards playing_guitar playing_trombone playing_bagpipes shoveling_snow
       bowling playing_violin chopping_wood "
OPTS+="--audio_path /home/cxu-serve/p1/ytian21/dat/Kinetics/data/audio "
OPTS+="--frame_path /home/cxu-serve/p1/ytian21/dat/Kinetics/data/frames "

OPTS+="--weights_Dv data/ckpt/Kinetics_av_reg-resnet18-anet-visual-learn-frames1stride8-maxpool-epoch30-step10_20 "
OPTS+="--weights_Da data/ckpt/Kinetics_av_reg-resnet18-anet-visual-learn-frames1stride8-maxpool-epoch30-step10_20 "

# weights
OPTS+="--weights_sound data/ckpt/Kinetics_av_reg-resnet18-anet-visual-learn-frames1stride8-maxpool-epoch30-step10_20/sound_latest.pth "
OPTS+="--weights_frame data/ckpt/Kinetics_av_reg-resnet18-anet-visual-learn-frames1stride8-maxpool-epoch30-step10_20/frame_latest.pth "
OPTS+="--weights_classifier data/ckpt/Kinetics_av_reg-resnet18-anet-visual-learn-frames1stride8-maxpool-epoch30-step10_20/classifier_latest.pth "

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
OPTS+="--num_gpus 4 "
OPTS+="--workers 32 "
OPTS+="--batch_size_per_gpu 8 "
OPTS+="--lr_frame 1e-4 " #1e-4
OPTS+="--lr_sound 1e-4 " #1e-3
OPTS+="--lr_classifier 1e-4 " #1e-3
OPTS+="--num_epoch 30 "
OPTS+="--lr_steps 10 20 "

# display, viz
OPTS+="--disp_iter 5 "

python -u main_defense.py $OPTS
