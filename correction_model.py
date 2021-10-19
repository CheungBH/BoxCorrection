import torch
from torch import nn

bg_dim = 16
cls_dim = 128
box_dim = 32


class CorrectionNet:
    def __init__(self, class_num):
        self.n_classes = class_num
        self.cls_embedding = nn.Embedding(self.n_classes, cls_dim)
        self.bg_embedding = nn.Embedding(2, bg_dim)
        self.box_embedding = nn.Embedding(4, box_dim)
        self.image_project = nn.Conv2d()
        self.image_pooling = nn.AdaptiveAvgPool2d()
        self.instance_project = nn.Conv2d()
        self.instance_pooling = nn.AdaptiveAvgPool2d()

        self.forward_net = nn.Sequential(
            nn.Linear(in_dim, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, num_classes)
        )

    def forward(self, image_feature, instance_feature, class_possibility, bg_possibility, box_reg):
        cls = self.cls_embedding(class_possibility)
        bg = self.bg_embedding(bg_possibility)
        box = self.box_embedding(box_reg)

        img = self.image_pooling(self.image_project(image_feature)).flatten()
        instance = self.instance_pooling(self.instance_project(instance_feature)).flatten()

        feat = torch.cat([img, instance, cls, bg, box], dim=-1)

        out = self.forward_net(feat)
        return out


