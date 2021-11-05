
from correction_model import CorrectionNet
from correction_loader import Dataloader
import torch.nn.functional as F
import torch
from utils import generate_box_weight, _smooth_l1_loss
import time
from opt import opt
from logger import ErrorAnalysisLogger

dataset_path = opt.dataset_root
device = opt.device
# model_path = opt.loadModel
model_path = "weights/best_27_new.pth"
logger = ErrorAnalysisLogger(model_path)
cls_crit = F.cross_entropy

dataset = Dataloader(dataset_path)
loader = dataset.build_loader(batch_size=1, num_worker=opt.num_worker)
net = CorrectionNet(dataset.num_class)
net.load_state_dict(torch.load(model_path, map_location=opt.device))
if device != "cpu":
    net.cuda()

net.eval()
errors = []

loss_before = torch.zeros(1)
loss_after = torch.zeros(1)
for i, data in enumerate(loader):
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

    loss_after += out_reg_loss + out_cls_loss
    loss_before += in_reg_loss + in_cls_loss

    # fg = torch.argmax(cls_label[0]).tolist()
    errors.append([image_name[0], cls_label[0].tolist(), in_cls_loss.tolist(), in_reg_loss.tolist(), out_cls_loss.tolist(), out_reg_loss.tolist()])

logger.write_summarize(errors, (loss_before, loss_after))
