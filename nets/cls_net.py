import torch
import torch.nn as nn

class Classifier_Concat(nn.Module):
    def __init__(self, cls_num):
        super(Classifier_Concat, self).__init__()
        self.fc1 = nn.Linear(1024, cls_num)

    def forward(self, feat_img, feat_sound):
        feat = torch.cat((feat_img,  feat_sound), dim =-1)
        g = self.fc1(feat)
        return g
