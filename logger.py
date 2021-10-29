import os


class BatchLogger:
    def __init__(self, folder):
        excel_path = os.path.join("/".join(folder.split("/")[:-1]), "train_result.csv")
        if not os.path.exists(excel_path):
            self.file = open(excel_path, "w")
            self.write_title()
        else:
            self.file = open(excel_path, "a+")
        self.model_idx = folder.split("/")[-1]

    def write_title(self):
        title = "model_idx, batch_size, optimizer, lr, epoch, balance_ratio, loss\n"
        self.file.write(title)

    def write_results(self, bs, opti, lr, epo, br, loss):
        self.file.write("{},{},{},{},{},{},{}\n".format(self.model_idx, bs, opti, lr, epo, br, loss))
