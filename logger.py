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

    def write_title(self):
        title = "image name, objectness, output cls-loss, output reg-loss, input cls-loss, input reg-loss\n"
        self.file.write(title)

    # def write_results(self, item):
    #     im_name, fg, out_cls, out_reg, in_cls, in_reg = item
    #     self.file.write("{},{},{},{},{},{}\n".format(im_name, fg, out_cls, out_reg, in_cls, in_reg))

    def write_summarize(self, ls, loss=()):
        for item in ls:
            im_name, fg, out_cls, out_reg, in_cls, in_reg = item
            self.file.write("{},{},{},{},{},{}\n".format(im_name, fg, out_cls, out_reg, in_cls, in_reg))
        if loss:
            self.file.write("\nLoss Before, {}".format(loss[0]))
            self.file.write("\nLoss After, {}".format(loss[1]))


