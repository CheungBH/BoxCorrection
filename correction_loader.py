import h5py
import os
from torch.utils.data import Dataset
import torch

tensor = torch.Tensor


class CorrectionDataset(Dataset):
    def __init__(self, h5_folders):
        self.files = [os.path.join(h5_folders, file_name) for file_name in os.listdir(h5_folders)]

    def __getitem__(self, item):
        with h5py.File(self.files[item]) as f:
            boxes_label = f["boxes_label"].value
            cls_label = f["cls_label"].value
            image_feature = f["image_feature"].value
            instance_feature = f["instance_feature"].value
            cls_preds = f["cls_preds"].value
            boxes_preds = f["boxes_preds"].value
        return tensor(boxes_label), tensor(cls_label), tensor(image_feature), tensor(instance_feature), \
               tensor(cls_preds), tensor(boxes_preds)

    def __len__(self):
        return len(self.files)


class Dataloader:
    def __init__(self, h5_folders):
        self.dataset = CorrectionDataset(h5_folders)

    def build_loader(self, shuffle=False, batch_size=1, num_worker=1, pin_memory=True):
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker,
            pin_memory=pin_memory)


if __name__ == '__main__':
    # CD = CorrectionDataset("h5")
    # print(CD[0])
    h5_folder = "h5"
    loader = Dataloader(h5_folder).build_loader()
    for idx, (boxes_label, cls_label, image_feature, instance_feature, cls_preds, boxes_preds) in enumerate(loader):
        print(boxes_label)


