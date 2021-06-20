#!/bin/bash

OPTS=""
OPTS+="--id Kinetics_av_reg_maxSim "
OPTS+="--list_train data/train_kinetics.csv "
OPTS+="--list_val data/val_kinetics.csv "

# Models
OPTS+="--arch_frame resnet18 "
OPTS+="--arch_sound anet "
OPTS+="--arch_classifier concat "
OPTS+="--img_pool maxpool "
OPTS+="--dic learn "

OPTS+="--cls_num 27 "
OPTS+="--categories laughing playing_clarinet singing
       playing_harmonica playing_keyboard playing_xylophone playing_bass_guitar
       tapping_guitar playing_drums playing_piano ripping_paper playing_saxophone
       tickling playing_trumpet tapping_pen playing_organ tap_dancing playing_accordion
       blowing_nose shuffling_cards playing_guitar playing_trombone playing_bagpipes shoveling_snow
       bowling playing_violin chopping_wood "
OPTS+="--audio_path /home/cxu-serve/p1/ytian21/dat/Kinetics/data/audio "
OPTS+="--frame_path /home/cxu-serve/p1/ytian21/dat/Kinetics/data/frames "

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
OPTS+="--workers 16 "
OPTS+="--batch_size_per_gpu 12 "
OPTS+="--lr_frame 1e-4 "
OPTS+="--lr_sound 1e-3 "
OPTS+="--lr_classifier 1e-3 "
OPTS+="--num_epoch 30 "
OPTS+="--lr_steps 10 20 "

# display, viz
OPTS+="--disp_iter 5 "

python -u main_attack.py $OPTS
