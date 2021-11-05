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
        title = "model_idx, batch_size, optimizer, lr, epoch, balance_ratio, schedule, momentum, loss\n"
        self.file.write(title)

    def write_results(self, bs, opti, lr, epo, br, loss, sched, mom):
        self.file.write("{},{},{},{},{},{},{},{},{}\n".format(self.model_idx, bs, opti, lr, epo, br, sched, mom, loss))


class ErrorAnalysisLogger:
    def __init__(self, folder):
        excel_path = os.path.join("/".join(folder.split("/")[:-1]), "error_analysis.csv")
        if not os.path.exists(excel_path):
            self.file = open(excel_path, "w")
            self.write_title()
        else:
            self.file = open(excel_path, "a+")
        self.model_idx = folder.split("/")[-1]
        self.loss_title = ("Loss sum before", "Loss sum after", "loss reg before", "loss reg after",
                           "Loss cls before", "Loss cls after")

    def write_title(self):
        title = "image name, objectness, output cls-loss, output reg-loss, input cls-loss, input reg-loss\n"
        self.file.write(title)

    def write_summarize(self, ls, losses=()):
        for item in ls:
            im_name, fg, out_cls, out_reg, in_cls, in_reg = item
            self.file.write("{},{},{},{},{},{}\n".format(im_name, fg, out_cls, out_reg, in_cls, in_reg))
        self.file.write("\n\n")
        for idx in range(len(losses)):
            self.file.write("\n{}, {}".format(self.loss_title[idx], losses[idx]))

