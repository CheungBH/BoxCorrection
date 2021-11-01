import h5py
import os
from torch.utils.data import Dataset
import torch
import numpy as np
import random

tensor = torch.Tensor


class CorrectionDataset(Dataset):
    def __init__(self, h5_folders, balance_ratio=0):
        self.files = [os.path.join(h5_folders, file_name) for file_name in os.listdir(h5_folders)]
        self.boxes_pred, self.cls_pred, self.instance_feature, self.cls_label, self.boxes_label, self.idx = \
            [], [], [], [], [], []
        self.image_feature = {}
        for i, file in enumerate(self.files):
            with h5py.File(file) as f:
                self.boxes_pred.append(f["boxes_preds"].value)
                self.cls_label.append(f["cls_label"].value[0])
                self.image_feature[i] = tensor(f["image_feature"].value[0])
                self.instance_feature.append(f["instance_feature"].value)
                self.cls_pred.append(f["cls_preds"].value)
                self.boxes_label.append(f["boxes_label"].value[0])
                self.idx += 256 * [i]
        self.boxes_pred = tensor(self.merge_sample(self.boxes_pred))
        self.cls_pred = tensor(self.merge_sample(self.cls_pred))
        self.instance_feature = tensor(self.merge_sample(self.instance_feature))
        self.cls_label = tensor(self.merge_sample(self.cls_label))
        self.boxes_label = tensor(self.merge_sample(self.boxes_label))
        self.balance_ratio = balance_ratio
        self.num_class = self.cls_pred.size()[1] - 1
        if balance_ratio:
            self.sample_balance()

    def sample_balance(self):
        pos_sample = int(self.cls_label.sum())
        neg_sample = pos_sample * self.balance_ratio
        if pos_sample + neg_sample > len(self):
            return
        # sorted_idx = self.cls_label.sort()[1]
        pos_idx, neg_idx = torch.nonzero(self.cls_label > 0).view(-1), torch.nonzero(self.cls_label == 0).view(-1)
        selected_idx = random.sample(neg_idx.tolist(), neg_sample) + pos_idx.tolist()
        self.boxes_pred = self.boxes_pred[selected_idx]
        self.cls_pred = self.cls_pred[selected_idx]
        self.instance_feature = self.instance_feature[selected_idx]
        self.cls_label = self.cls_label[selected_idx]
        self.boxes_label = self.boxes_label[selected_idx]
        self.idx = torch.Tensor(self.idx)[selected_idx].tolist()

    @staticmethod
    def merge_sample(item):
        temp = item[0]
        for i in item[1:]:
            temp = np.concatenate((temp, i))
        return temp

    def __getitem__(self, item):
        idx = self.idx[item]
        boxes_label = self.boxes_label[item]
        cls_label = self.cls_label[item]
        image_feature = self.image_feature[idx]
        instance_feature = self.instance_feature[item]
        cls_preds = self.cls_pred[item]
        boxes_preds = self.boxes_pred[item]
        return boxes_label, cls_label, image_feature, instance_feature, boxes_preds, cls_preds

    def __len__(self):
        return len(self.idx)


class Dataloader:
    def __init__(self, h5_folders, balance_ratio=0):
        self.dataset = CorrectionDataset(h5_folders, balance_ratio=balance_ratio)
        self.num_class = self.dataset.num_class

    def build_loader(self, shuffle=False, batch_size=1, num_worker=1, pin_memory=True):
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker,
            pin_memory=pin_memory)


if __name__ == '__main__':
    # CD = CorrectionDataset("h5")
    # print(CD[0])
    h5_folder = "h5/fake_sim10k"
    loader = Dataloader(h5_folder).build_loader(num_worker=0)
    for idx, (boxes_label, cls_label, image_feature, instance_feature, cls_preds, boxes_preds) in enumerate(loader):
        print(idx)


