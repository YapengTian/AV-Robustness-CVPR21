import torch
import torchvision
import torch.nn.functional as F

from .audio_net import  ANet
from .vision_net import  Resnet
from .cls_net import Classifier_Concat
from .criterion import BCELoss, CELoss


def activate(x, activation):
    if activation == 'sigmoid':
        return torch.sigmoid(x)
    elif activation == 'softmax':
        return F.softmax(x, dim=1)
    elif activation == 'relu':
        return F.relu(x)
    elif activation == 'tanh':
        return F.tanh(x)
    elif activation == 'no':
        return x
    else:
        raise Exception('Unkown activation!')


class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)

    def build_sound(self, arch='unet6', weights=''):

        if arch == "anet":
            net_sound = ANet()
        else:
            raise Exception('Architecture undefined!')

        #net_sound.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_sound')
            net_sound.load_state_dict(torch.load(weights))

        return net_sound

    # builder for vision
    def build_frame(self, arch='resnet18', pool_type='avgpool',
                    weights=''):
        pretrained=True
        if arch == 'resnet18':
            original_resnet = torchvision.models.resnet18(pretrained)
            net = Resnet(
                original_resnet, pool_type=pool_type)
        else:
            raise Exception('Architecture undefined!')

        if len(weights) > 0:
            print('Loading weights for net_frame')
            net.load_state_dict(torch.load(weights))
        return net

    def build_classifier(self, arch, cls_num, weights=''):
        if arch == 'concat':
            net = Classifier_Concat(cls_num)
        else:
            raise Exception('Architecture undefined!')

        net.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_grounding')
            net.load_state_dict(torch.load(weights))
        return net


    def build_criterion(self, arch):
        if arch == 'bce':
            net = BCELoss()
        elif arch == 'ce':
            net = CELoss()
        else:
            raise Exception('Architecture undefined!')
        return net
