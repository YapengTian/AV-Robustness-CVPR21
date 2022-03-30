#!/bin/bash

OPTS=""
OPTS+="--mode eval "
OPTS+="--id AVE_av_reg_defense-resnet18-anet-concat-fix-frames1stride8-maxpool-epoch30-step10_20 "
OPTS+="--list_train data/train_AVE.csv "
OPTS+="--list_val data/test_AVE.csv "

# Models
OPTS+="--arch_frame resnet18 "
OPTS+="--arch_sound anet "
OPTS+="--arch_classifier concat "
OPTS+="--img_pool maxpool "
OPTS+="--dic fix "


OPTS+="--cls_num 28 "
OPTS+="--categories Church_bell Male_speech Bark airplane Race_car Female_speech Helicopter Violin Flute Ukulele Frying
                    Truck Shofar Motorcycle Chainsaw Acoustic_guitar Train_horn Clock Banjo Goat Baby_cry Bus Cat Horse Toilet_flush Rodents
                    Accordion Mandolin "
OPTS+="--audio_path /home/cxu-serve/p1/ytian21/dat/AVSS_data/AVE_Dataset/data/audio "
OPTS+="--frame_path /home/cxu-serve/p1/ytian21/dat/AVSS_data/AVE_Dataset/data/frames "

OPTS+="--weights_Dv data/ckpt/AVE_av_reg-resnet18-anet-concat-learn-frames1stride8-maxpool-epoch30-step10_20 "
OPTS+="--weights_Da data/ckpt/AVE_av_reg-resnet18-anet-concat-learn-frames1stride8-maxpool-epoch30-step10_20 "

# attack type
OPTS+="--attack_type mim "

# logscale in frequency
OPTS+="--num_mix 1 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--num_frames 1 "
OPTS+="--stride_frames 8 "
OPTS+="--frameRate 8 "

# Sparse solver
OPTS+="--alpha_a  7e-7 "
OPTS+="--alpha_v  2e-6 "

# audio-related
OPTS+="--audLen 65535 " # 65535
OPTS+="--audRate 11025 " #11025

OPTS+="--num_gpus 2 "
OPTS+="--batch_size_per_gpu 4 "

python -u main_defense.py $OPTS