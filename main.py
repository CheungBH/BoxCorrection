
from correction_model import CorrectionNet
from correction_loader import Dataloader

epochs = 1
dataset_path = "h5"

net = CorrectionNet(1)
loader = Dataloader(dataset_path).build_loader(batch_size=2, num_worker=0)


for epoch in range(epochs):
    for i, data in enumerate(loader):
        boxes_label, cls_label, image_feature, instance_feature, boxes_preds, cls_preds = data
        output = net(image_feature, instance_feature, cls_preds, boxes_preds)
        print(output)

