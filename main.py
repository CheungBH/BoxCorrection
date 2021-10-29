
from correction_model import CorrectionNet
from correction_loader import Dataloader
import torch.nn.functional as F
import torch.optim as optim
import torch
import os
from utils import generate_box_weight, _smooth_l1_loss

try:
    from apex import amp
    mix_precision = True
except:
    mix_precision = False

epochs = 20
dataset_path = "h5"
LR = 0.0000005
device = "cpu"
model_dir = "5E-7_BN"
os.makedirs(model_dir, exist_ok=True)

cls_crit = F.cross_entropy

net = CorrectionNet(1)
loader = Dataloader(dataset_path).build_loader(batch_size=2, num_worker=0)
optimizer = optim.Adam(net.parameters(), lr=LR)
iteration = 0
log = open(os.path.join(model_dir, "log.txt"), "w")


if device != "cpu":
    net.cuda()

for epoch in range(epochs):
    net.train()
    print("Processing epoch {}".format(epoch))
    loss_sum = torch.zeros(1)
    if device != "cpu":
        loss_sum = loss_sum.cuda()

    for i, data in enumerate(loader):
        iteration += 1
        boxes_label, cls_label, image_feature, instance_feature, boxes_preds, cls_preds = data
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
    print("Average loss is {}".format(ave_loss))
    log.write("Epoch {}: Loss {}".format(epoch, ave_loss))
    torch.save(net.state_dict(), os.path.join(model_dir, "{}.pth".format(epoch)))

