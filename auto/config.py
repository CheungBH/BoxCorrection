#-*-coding:utf-8-*-
excel_path = "auto/balance_sample/ratio3.csv"
excel_path = excel_path.replace("\\", "/")

task_folder = excel_path.split("/")[-2]
excel_name = excel_path.split("/")[-1]
base_name = excel_name.split(".")[0]

computer = "desktop"
