import h5py


class CorrectionDataset:
    def __init__(self, file):
        with h5py.File(file) as f:
            self.boxes = f["boxes"].value
            self.cls = f["classification"].value
            self.image_feature = f["image_feature"]
            self.instance_feature = f["instance_feature"]

        

if __name__ == '__main__':
    CD = CorrectionDataset("dataset.h5")
    print(CD.boxes)

