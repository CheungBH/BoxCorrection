import torch
from torch import nn

cls_dim = 16
box_dim = 32


class CorrectionNet(nn.Module):
    def __init__(self, class_num):
        super(CorrectionNet, self).__init__()
        self.n_classes = class_num
        self.cls_embedding = nn.Linear(self.n_classes+1, cls_dim)
        self.box_embedding = nn.Linear(4, box_dim)
        self.image_project = nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=3, bias=False)
        self.image_pooling = nn.AdaptiveAvgPool2d(1)
        self.instance_project = nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=3, bias=False)
        self.instance_pooling = nn.AdaptiveAvgPool2d(1)
        self.hdim = 128

        self.forward_net = nn.Sequential(
            nn.Linear(176, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, class_num+5)
        )

    def forward(self, image_feature, instance_feature, class_possibility, box_reg):
        cls = self.cls_embedding(class_possibility)
        box = self.box_embedding(box_reg)

        img = self.image_pooling(self.image_project(image_feature)).view(1,-1)
        instance = self.instance_pooling(self.instance_project(instance_feature)).view(1,-1)

        feat = torch.cat([img, instance, cls, box], dim=-1)

        out = self.forward_net(feat)
        return out


if __name__ == '__main__':
    from correction_loader import CorrectionDataset
    folder = "h5"
    boxes_label, cls_label, image_feature, instance_feature, cls_preds, boxes_preds = CorrectionDataset(folder)[0]
    net = CorrectionNet(class_num=1)
    output = net(image_feature, instance_feature[0].unsqueeze(dim=0), cls_preds[0].unsqueeze(dim=0), boxes_preds[0].unsqueeze(dim=0))
    print(output)

