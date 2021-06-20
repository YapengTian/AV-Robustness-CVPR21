#!/bin/bash

OPTS=""
OPTS+="--id AVE_av_reg_mism_FB "
OPTS+="--list_train data/train_AVE.csv "
OPTS+="--list_val data/val_AVE.csv "

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

OPTS+="--weights_Dv data/ckpt/AVE_av_reg_msim-resnet18-anet-concat-learn-frames1stride8-maxpool-epoch30-step10_20 "
OPTS+="--weights_Da data/ckpt/AVE_av_reg_msim-resnet18-anet-concat-learn-frames1stride8-maxpool-epoch30-step10_20 "

# weights
OPTS+="--weights_sound data/ckpt/AVE_av_reg_msim-resnet18-anet-concat-learn-frames1stride8-maxpool-epoch30-step10_20/sound_latest.pth "
OPTS+="--weights_frame data/ckpt/AVE_av_reg_msim-resnet18-anet-concat-learn-frames1stride8-maxpool-epoch30-step10_20/frame_latest.pth "
OPTS+="--weights_classifier data/ckpt/AVE_av_reg_msim-resnet18-anet-concat-learn-frames1stride8-maxpool-epoch30-step10_20/classifier_latest.pth "


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
OPTS+="--batch_size_per_gpu 8 "
OPTS+="--lr_frame 1e-4 "
OPTS+="--lr_sound 1e-3 "
OPTS+="--lr_classifier 1e-3 "
OPTS+="--num_epoch 30 "
OPTS+="--lr_steps 10 20 "

# display, viz
OPTS+="--disp_iter 5 "

python -u main_defense.py $OPTS
