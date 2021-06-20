import shutil
import subprocess
import os
import argparse
import glob

def extract_frames(video, dst):
    command1 = 'ffmpeg '
    command1 += '-i ' + video + " "
    command1 += '-y' + " "
    command1 += "-r " + "8 "
    command1 += '{0}/%06d.jpg'.format(dst)
    print(command1)
    #    print command1
    os.system(command1)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', dest='out_dir', type=str, default='/home/cxu-serve/p1/ytian21/dat/AVSS_data/AVE_Dataset/data/frames')
    parser.add_argument('--video_path', dest='video_path', type=str, default='/home/cxu-serve/p1/ytian21/dat/AVSS_data/AVE_Dataset/AVE_video/')
    args = parser.parse_args()

    vid_list = os.listdir(args.video_path)

    for vid in vid_list:
        vid_pth = os.path.join(args.video_path, vid)
        lis = os.listdir(vid_pth)
        if not os.path.exists(os.path.join(args.out_dir, vid)):
            os.makedirs(os.path.join(args.out_dir, vid))
        print(vid)
        if vid != "Frying":
            continue
        for vid_id in lis:
            if vid_id[-4:] != ".mp4":
                continue
            name = os.path.join(args.video_path, vid, vid_id)
            dst = os.path.join(args.out_dir, vid, vid_id)
            print(dst)
            if not os.path.exists(dst):
                os.makedirs(dst)
            extract_frames(name, dst)
            print("finish video id: " + vid_id)

