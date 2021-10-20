import torch
from torch import nn

cls_dim = 128
box_dim = 32


class CorrectionNet(nn.Module):
    def __init__(self, class_num):
        super(CorrectionNet, self).__init__()
        self.n_classes = class_num
        self.cls_embedding = nn.Linear(self.n_classes+1, cls_dim)
        self.box_embedding = nn.Linear(4, box_dim)
        self.image_project = nn.Conv2d()
        self.image_pooling = nn.AdaptiveAvgPool2d()
        self.instance_project = nn.Conv2d()
        self.instance_pooling = nn.AdaptiveAvgPool2d()
        self.hdim = 128

        self.forward_net = nn.Sequential(
            nn.Linear(in_dim, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, class_num)
        )

    def forward(self, image_feature, instance_feature, class_possibility, box_reg):
        cls = self.cls_embedding(class_possibility)
        box = self.box_embedding(box_reg)

        img = self.image_pooling(self.image_project(image_feature)).flatten()
        instance = self.instance_pooling(self.instance_project(instance_feature)).flatten()

        feat = torch.cat([img, instance, cls, box], dim=-1)

        out = self.forward_net(feat)
        return out


if __name__ == '__main__':
    from correction_loader import CorrectionDataset
    folder = "h5"
    boxes_label, cls_label, image_feature, instance_feature, cls_preds, boxes_preds = CorrectionDataset(folder)[0]
    net = CorrectionNet(class_num=1)
    output = net(image_feature, instance_feature, cls_preds, boxes_preds)

