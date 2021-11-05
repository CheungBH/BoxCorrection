
from correction_model import CorrectionNet
from correction_loader import Dataloader
import torch.nn.functional as F
import torch
import os
from utils import generate_box_weight, _smooth_l1_loss
import time
from opt import opt

dataset_path = opt.dataset_root
device = opt.device
model_dir = os.path.join("weights", opt.expFolder, opt.expID,)
optimize = opt.optMethod
os.makedirs(model_dir, exist_ok=True)

cls_crit = F.cross_entropy

dataset = Dataloader(dataset_path)
loader = dataset.build_loader(batch_size=1, num_worker=opt.num_worker)
net = CorrectionNet(dataset.num_class)
if device != "cpu":
    net.cuda()

log = open(os.path.join(model_dir, "log.txt"), "w")
best_loss = float("inf")

net.eval()

loss_sum = torch.zeros(1)
errors = []
if device != "cpu":
    loss_sum = loss_sum.cuda()

for i, data in enumerate(loader):
    opt.iterations += 1
    boxes_label, cls_label, image_feature, instance_feature, boxes_preds, cls_preds, image_name = data
    if device != "cpu":
        output = net(image_feature.cuda(), instance_feature.cuda(), cls_preds.cuda(), boxes_preds.cuda()).cpu()
    else:
        output = net(image_feature, instance_feature, cls_preds, boxes_preds)

    b_weight = generate_box_weight(cls_label)
    out_cls_loss = cls_crit(output[:, :2], cls_label.long())
    out_reg_loss = _smooth_l1_loss(output[:, 2:], boxes_label, b_weight, b_weight)
    in_cls_loss = cls_crit(cls_preds, cls_label.long())
    in_reg_loss = _smooth_l1_loss(boxes_preds, boxes_label, b_weight, b_weight)

    fg = torch.argmax(cls_label[0]).tolist()
    errors.append([image_name, fg, in_cls_loss, in_reg_loss, out_cls_loss, out_reg_loss])

print(errors)

