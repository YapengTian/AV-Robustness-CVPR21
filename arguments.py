import argparse


class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        # Model related arguments
        parser.add_argument('--id', default='',
                            help="a name for identifying the model")
        parser.add_argument('--num_mix', default=1, type=int,
                            help="number of sounds to mix")
        parser.add_argument('--arch_sound', default='vggish',
                            help="architecture of net_sound")
        parser.add_argument('--arch_frame', default='resnet18',
                            help="architecture of net_frame")
        parser.add_argument('--arch_classifier', default='base',
                            help="architecture of net_classifier")
        parser.add_argument('--weights_sound', default='',
                            help="weights to finetune net_sound")
        parser.add_argument('--weights_frame', default='',
                            help="weights to finetune net_frame")
        parser.add_argument('--weights_memory', default='',
                            help="weights to finetune net_memory")
        parser.add_argument('--weights_classifier', default='',
                            help="weights to finetune net_classifier")
        parser.add_argument('--weights_Dv', default='',
                            help="visual Dic")
        parser.add_argument('--weights_Da', default='',
                            help="audio Dic")
        parser.add_argument('--train_mode', default='adv',
                            help="adversarial training")
        parser.add_argument('--num_channels', default=32, type=int,
                            help='number of channels')
        parser.add_argument('--num_frames', default=1, type=int,
                            help='number of frames')
        parser.add_argument('--stride_frames', default=1, type=int,
                            help='sampling stride of frames')
        parser.add_argument('--img_pool', default='maxpool',
                            help="avg or max pool image features")
        parser.add_argument('--img_activation', default='no',
                            help="activation on the image features")
        parser.add_argument('--sound_activation', default='no',
                            help="activation on the sound features")
        parser.add_argument('--output_activation', default='sigmoid',
                            help="activation on the output")
        parser.add_argument('--attack_type', default='fsgm',
                            help="attack type")
        parser.add_argument('--loss', default='ce',
                            help="loss function to use")
        parser.add_argument('--dic', default='learn',
                            help="dic")
        parser.add_argument('--log_freq', default=1, type=int,
                            help="log frequency scale")

        # Data related arguments
        parser.add_argument('--num_gpus', default=1, type=int,
                            help='number of gpus to use')
        parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                            help='input batch size')
        parser.add_argument('--workers', default=16, type=int,
                            help='number of data loading workers')
        parser.add_argument('--num_val', default=-1, type=int,
                            help='number of images to evalutate')
        parser.add_argument('--num_vis', default=40, type=int,
                            help='number of images to evalutate')

        parser.add_argument('--audLen', default=65535, type=int,
                            help='sound length')
        parser.add_argument('--audRate', default=11025, type=int,
                            help='sound sampling rate')
        parser.add_argument('--stft_frame', default=1022, type=int,
                            help="stft frame length")
        parser.add_argument('--stft_hop', default=256, type=int,
                            help="stft hop length")

        parser.add_argument('--imgSize', default=224, type=int,
                            help='size of input frame')
        parser.add_argument('--frameRate', default=8, type=float,
                            help='video frame sampling rate')

        # Misc arguments
        parser.add_argument('--seed', default=1234, type=int,
                            help='manual seed')
        parser.add_argument('--ckpt', default='data/ckpt',
                            help='folder to output checkpoints')
        parser.add_argument('--disp_iter', type=int, default=20,
                            help='frequency to display')
        parser.add_argument('--eval_epoch', type=int, default=1,
                            help='frequency to evaluate')
        # data
        parser.add_argument('--cls_num', default=11, type=int,
                            help='total class numbers')
        parser.add_argument('--audio_path', default='/home/cxu-serve/p1/ytian21/dat/AVSS_data/MUSIC_dataset/data/audio')
        parser.add_argument('--frame_path', default='/home/cxu-serve/p1/ytian21/dat/AVSS_data/MUSIC_dataset/data/frames_8fps')
        parser.add_argument("--categories", nargs="+", default=
        ['flute','acoustic_guitar','accordion','xylophone','erhu', 'tuba','saxophone','cello','violin','clarinet','trumpet'])

        self.parser = parser

    def add_train_arguments(self):
        parser = self.parser

        parser.add_argument('--mode', default='train',
                            help="train/eval")
        parser.add_argument('--list_train',
                            default='data/train.csv')
        parser.add_argument('--list_val',
                            default='data/val.csv')


        # optimization related arguments
        parser.add_argument('--num_epoch', default=100, type=int,
                            help='epochs to train for')
        parser.add_argument('--lr_frame', default=1e-4, type=float, help='LR')
        parser.add_argument('--lr_sound', default=1e-3, type=float, help='LR')
        parser.add_argument('--lr_prox', default=0.1, type=float, help='LR')
        parser.add_argument('--lr_conprox', default=1e-4, type=float, help='LR')
        parser.add_argument('--lr_classifier', default=1e-3, type=float, help='LR')
        parser.add_argument('--lr_steps', nargs='+', type=int, default=[40, 60],
                            help='steps to drop LR in epochs')
        parser.add_argument('--beta1', default=0.9, type=float,
                            help='momentum for sgd, beta1 for adam')
        parser.add_argument('--weight_decay', default=1e-4, type=float,
                            help='weights regularizer')
        self.parser = parser

    def print_arguments(self, args):
        print("Input arguments:")
        for key, val in vars(args).items():
            print("{:16} {}".format(key, val))

    def parse_train_arguments(self):
        self.add_train_arguments()
        args = self.parser.parse_args()
        self.print_arguments(args)
        return args
