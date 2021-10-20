
from correction_model import CorrectionNet
from correction_loader import Dataloader

epochs = 20
dataset_path = "h5"

net = CorrectionNet()
loader = Dataloader(dataset_path).build_loader()


for epoch in range(epochs):
    for i, data in enumerate(loader):
        boxes_label, cls_label, image_feature, instance_feature, cls_preds, boxes_preds = data





