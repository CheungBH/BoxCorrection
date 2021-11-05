import os


class BaseLogger:
    def __init__(self, folder):
        self.folder = folder
        self.model_idx = self.folder.split("/")[-1]
        self.title = ""

    def init(self, kw):
        excel_path = os.path.join("/".join(self.folder.split("/")[:-1]), "{}.csv".format(kw))
        if not os.path.exists(excel_path):
            self.file = open(excel_path, "w")
            self.write_title()
        else:
            self.file = open(excel_path, "a+")

    @staticmethod
    def list2str(ls):
        tmp = ""
        for item in ls:
            if isinstance(item, str):
                tmp += item
            else:
                tmp += str(round(item, 4))
            tmp += ","
        return tmp[:-1]

    def write_title(self):
        self.file.write(self.title)

    def write_summarize(self, ls):
        for item in ls:
            self.file.write("{}\n".format(self.list2str(item)))
        self.file.write("\n")


class TrainBatchLogger(BaseLogger):
    def __init__(self, folder):
        super(TrainBatchLogger, self).__init__(folder)
        self.title = "model_idx, batch_size, optimizer, lr, epoch, balance_ratio, schedule, momentum, loss\n"
        self.init("train_result")


class ErrorAnalysisLogger(BaseLogger):
    def __init__(self, folder):
        super(ErrorAnalysisLogger, self).__init__(folder)
        self.title = "image name,objectness,output cls-loss,output reg-loss,input cls-loss,input reg-loss\n"
        self.init("error_analysis")


class ErrorSummaryLogger(BaseLogger):
    def __init__(self, folder):
        super(ErrorSummaryLogger, self).__init__(folder)
        self.title = "image name, positive num, negative num, input loss, output loss, input-cls loss, " \
                     "output-cls loss, input-reg loss, output-reg loss, fg input-cls loss, fg output-cls loss, " \
                     "bg input-cls loss, bg output-cls loss\n"
        self.init("error_summary")


