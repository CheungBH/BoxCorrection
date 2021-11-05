
import pandas as pd
from logger import ErrorSummaryLogger


def generate_boundary(unit, times):
    ls, sum_tmp = [0], 0
    for i in range(times):
        sum_tmp += unit
        ls.append(sum_tmp)
    return ls


def analyse(df, name="all"):
    # image_names = set
    valid_anchor = df["objectness"].tolist()
    positive_num = sum(valid_anchor)
    negative_num = len(valid_anchor) - positive_num
    loss_cls_before = sum(df["input cls-loss"]) / len(valid_anchor)
    loss_reg_before = sum(df["input reg-loss"]) / len(valid_anchor)
    loss_cls_after = sum(df["output cls-loss"]) / len(valid_anchor)
    loss_reg_after = sum(df["output reg-loss"]) / len(valid_anchor)
    loss_before = loss_cls_before + loss_reg_before
    loss_after = loss_reg_after + loss_cls_after

    loss_cls_fg_before = sum(df["input cls-loss"]*df["objectness"])/positive_num
    loss_cls_fg_after = sum(df["output cls-loss"]*df["objectness"])/positive_num
    loss_cls_bg_before = sum(df["input cls-loss"]*(1-df["objectness"]))/negative_num
    loss_cls_bg_after = sum(df["output cls-loss"]*(1-df["objectness"]))/negative_num
    return [name, positive_num, negative_num, loss_before, loss_after, loss_cls_before, loss_cls_after, loss_reg_before,
            loss_reg_after, loss_cls_fg_before, loss_cls_fg_after, loss_cls_bg_before, loss_cls_bg_after]


if __name__ == '__main__':
    csv_path = 'weights/error_analysis.csv'
    content = pd.read_csv(csv_path)
    image_num = len(set(content["image name"].tolist()))
    anchor_per_img = int(len(content) / image_num)
    image_boundaries = generate_boundary(anchor_per_img, image_num)

    summary_ls = [analyse(content)]

    for idx in range(image_num):
        summary_ls.append(analyse(content[image_boundaries[idx]: image_boundaries[idx + 1]],
                                  content["image name"][image_boundaries[idx]]))

    print(summary_ls)
    summary_logger = ErrorSummaryLogger(csv_path)
    summary_logger.write_summarize(summary_ls)
