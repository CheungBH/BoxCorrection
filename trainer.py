import os

from correction_model import CorrectionNet
from correction_loader import Dataloader
import torch.nn.functional as F
import torch.optim as optim
import torch
import os
from utils import generate_box_weight, _smooth_l1_loss
import time
from opt import opt


class Trainer:
    def __init__(self, opt):
        self.epochs = opt.nEpochs
        self.dataset_path = opt.dataset_root
        self.LR = opt.LR
        self.device = opt.device
        self.model_dir = os.path.join("weights", opt.expFolder, opt.expID, )
        self.optimize = opt.optMethod
        os.makedirs(self.model_dir, exist_ok=True)

        self.cls_crit = F.cross_entropy

        dataset = Dataloader(opt.dataset_root, opt.balance_ratio)
        self.loader = dataset.build_loader(batch_size=opt.batch_size, num_worker=opt.num_worker)
        self.net = CorrectionNet(dataset.num_class)

        if opt.optMethod == "adam":
            self.optimizer = optim.Adam(self.net.parameters(), lr=opt.LR)
        elif opt.optMethod == "sgd":
            self.optimizer = optim.SGD(self.net.parameters(), lr=opt.LR)
        self.iteration = 0
        self.log = open(os.path.join(self.model_dir, "log.txt"), "w")

        if self.device != "cpu":
            self.net.cuda()
        net.train()


