
from correction_model import CorrectionNet
from correction_loader import Dataloader
import torch.nn.functional as F

epochs = 1
dataset_path = "h5"
device = "cpu"

cls_crit = F.cross_entropy

net = CorrectionNet(1)
loader = Dataloader(dataset_path).build_loader(batch_size=2, num_worker=0)

if device != "cpu":
    net.cuda()

for epoch in range(epochs):
    net.train()
    for i, data in enumerate(loader):
        boxes_label, cls_label, image_feature, instance_feature, boxes_preds, cls_preds = data
        output = net(image_feature, instance_feature, cls_preds, boxes_preds)
        # print(output)
        crit_loss = cls_crit(output[:, :2], cls_label)
        print(crit_loss)


