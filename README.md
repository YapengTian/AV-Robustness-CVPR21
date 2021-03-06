Can audio-visual integration strengthen robustness under multimodal attacks? (To appear in CVPR 2021) [[Paper]](https://arxiv.org/pdf/2104.02000.pdf)

[Yapeng Tian](http://yapengtian.org/) and [Chenliang Xu](https://www.cs.rochester.edu/~cxu22/) 


### Robustness of audio-visual learning under multimodal attacks

we propose to make a systematic study on machines’ multisensory perception under attacks. We
use the audio-visual event recognition task against multimodal adversarial attacks as a proxy to investigate the robustness of audio-visual learning. We attack audio, visual,
and both modalities to explore whether audio-visual integration still strengthens perception and how different fusion
mechanisms affect the robustness of audio-visual models.

![image](doc/attack_fig.png)

### Requirements

```bash
pip install -r requirements
```


### Datasets
1. Prepare video datasets.

    a. Download AVE dataset from: https://drive.google.com/file/d/1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK/view

    b. Download MUSIC dataset from: https://github.com/roudimit/MUSIC_dataset/blob/master/MUSIC_solo_videos.json
    
    c. Download Kinetics-Sound (KS) dataset. KS is a subset of [Kinetics dataset](https://github.com/cvdfoundation/kinetics-dataset) in 27 categories: laughing, playing_clarinet, singing,
       playing_harmonica, playing_keyboard, playing_xylophone, playing_bass_guitar,
       tapping_guitar, playing_drums, playing_piano, ripping_paper, playing_saxophone,
       tickling, playing_trumpet, tapping_pen, playing_organ, tap_dancing, playing_accordion,
       blowing_nose, shuffling_cards, playing_guitar, playing_trombone, playing_bagpipes, shoveling_snow,
       bowling, playing_violin, chopping_wood.

2. Preprocess videos. Please check scripts: [scripts/extract_audio.py](https://github.com/YapengTian/AV-Robustness-CVPR21/blob/master/scripts/extract_audio.py) and [scripts/extract_frames.py](https://github.com/YapengTian/AV-Robustness-CVPR21/blob/master/scripts/extract_frames.py). 

    a. Extract frames at 8fps and waveforms at 11025Hz from videos. We have following directory structure for each dataset:
    ```  
    ├── MUSIC
    ├── data
    ├── audio
    |   ├── acoustic_guitar
    │   |   ├── M3dekVSwNjY.wav
    │   |   ├── ...
    │   ├── trumpet
    │   |   ├── STKXyBGSGyE.wav
    │   |   ├── ...
    │   ├── ...
    |
    └── frames
    |   ├── acoustic_guitar
    │   |   ├── M3dekVSwNjY.mp4
    │   |   |   ├── 000001.jpg
    │   |   |   ├── ...
    │   |   ├── ...
    │   ├── trumpet
    │   |   ├── STKXyBGSGyE.mp4
    │   |   |   ├── 000001.jpg
    │   |   |   ├── ...
    │   |   ├── ...
    │   ├── ...
    ```




### Multimodal Attack
There are typos in the captions of Tab. 1 and Tab. 3. The perturbation strengthens are 0.006 and 0.012 as in Fig. 3 (10^-3) rather than 0.06 and 0.12. Next, we will use experiments on the AVE dataset as an example. Scripts for other datasets can be found in [scripts](https://github.com/YapengTian/AV-Robustness-CVPR21/tree/master/scripts).

Training:

```bash
./scripts/train_attack_AVE.sh
```

Testing: 

```bash
./scripts/eval_attack_AVE.sh
```

### Audio-Visual Defense against Multimodal Attacks 

In our original defense experiments, we learn the external feature memory banks using the MinSim constraint-based model. In implementation, we can add the MinSim loss term (please see L468 of main_attack.py) to re-train the above model. Then, run the following defense training.
Since the MUSIC dataset is small, it is important to follow our original training pipeline to reproduce our results. For the AVE dataset, directly training the defense model can obtain comparable results as shown in the supp of the paper. 

Training:

```bash
./scripts/train_defense_AVE.sh
```

Testing: 

```bash
./scripts/eval_defense_AVE.sh
```

### Citation

If you find this work useful, please consider citing it.

<pre><code>
@inproceedings{tian2021can,
  title={Can audio-visual integration strengthen robustness under multimodal attacks?},
  author={Tian, Yapeng and Xu, Chenliang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5601--5611},
  year={2021}
}
</code></pre>


 
 ### Acknowledgement
 
We borrowed a lot of code from [SoP](https://github.com/hangzhaomit/Sound-of-Pixels). We thank the authors for sharing their codes. If you use our code, please also cite their nice work.
