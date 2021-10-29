
from correction_model import CorrectionNet
from correction_loader import Dataloader
import torch.nn.functional as F
import torch.optim as optim
import torch
import os
from utils import generate_box_weight, _smooth_l1_loss
import time
from opt import opt

try:
    from apex import amp
    mix_precision = False
except:
    mix_precision = False

epochs = opt.nEpochs
dataset_path = opt.dataset_root
LR = opt.LR
device = opt.device
model_dir = os.path.join("weights", opt.expFolder, opt.expID,)
optimize = opt.optMethod
balance_ratio = opt.balance_ratio
os.makedirs(model_dir, exist_ok=True)

cls_crit = F.cross_entropy

dataset = Dataloader(dataset_path, balance_ratio)
loader = dataset.build_loader(batch_size=opt.batch_size, num_worker=opt.num_worker)
net = CorrectionNet(dataset.num_class)

if optimize == "adam":
    optimizer = optim.Adam(net.parameters(), lr=LR)
elif optimize == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=LR)
iteration = 0
log = open(os.path.join(model_dir, "log.txt"), "w")


if device != "cpu":
    net.cuda()
net.train()

for epoch in range(epochs):
    loss_sum = torch.zeros(1)
    begin_time = time.time()
    if device != "cpu":
        loss_sum = loss_sum.cuda()

    for i, data in enumerate(loader):
        iteration += 1
        boxes_label, cls_label, image_feature, instance_feature, boxes_preds, cls_preds = data
        if device != "cpu":
            output = net(image_feature.cuda(), instance_feature.cuda(), cls_preds.cuda(), boxes_preds.cuda()).cpu()
        else:
            output = net(image_feature, instance_feature, cls_preds, boxes_preds)

        cls_loss = cls_crit(output[:, :2], cls_label.long())
        b_weight = generate_box_weight(cls_label)
        reg_loss = _smooth_l1_loss(output[:, 2:], boxes_label, b_weight, b_weight)
        loss = cls_loss + reg_loss
        loss_sum += loss

        if mix_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

    ave_loss = (loss_sum/len(loader)).tolist()[0]
    print("The Average loss of epoch {} is {}, using {} s".format(epoch, ave_loss, round(time.time() - begin_time), 2))
    log.write("Epoch {}: Loss {}\n".format(epoch, ave_loss))
    torch.save(net.state_dict(), os.path.join(model_dir, "{}.pth".format(epoch)))

