
cmds = [
    "python main.py --LR 2E-7 --expFolder first_test --expID 1 --optMethod adam --balance_ratio 3",
    "python main.py --LR 2E-7 --expFolder first_test --expID 2 --optMethod sgd --balance_ratio 3"
]

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)
