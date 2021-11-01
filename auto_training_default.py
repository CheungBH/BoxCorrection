
cmds = [
'python main.py --batch_size 2 --optMethod rmsprop --epochs 200 --LR 1.00E-07 --balance_ratio 3 --schedule step --momentum 0 --expFolder balance_sample-ratio3 --expID 1',
'python main.py --batch_size 2 --optMethod adam --epochs 200 --LR 1.00E-07 --balance_ratio 3 --schedule step --momentum 0 --expFolder balance_sample-ratio3 --expID 2',
'python main.py --batch_size 2 --optMethod sgd --epochs 200 --LR 1.00E-07 --balance_ratio 3 --schedule step --momentum 0 --expFolder balance_sample-ratio3 --expID 3',
'python main.py --batch_size 2 --optMethod rmsprop --epochs 200 --LR 2.00E-07 --balance_ratio 3 --schedule step --momentum 0 --expFolder balance_sample-ratio3 --expID 4',
'python main.py --batch_size 2 --optMethod adam --epochs 200 --LR 2.00E-07 --balance_ratio 3 --schedule step --momentum 0 --expFolder balance_sample-ratio3 --expID 5',
'python main.py --batch_size 2 --optMethod sgd --epochs 200 --LR 2.00E-07 --balance_ratio 3 --schedule step --momentum 0 --expFolder balance_sample-ratio3 --expID 6',
'python main.py --batch_size 2 --optMethod rmsprop --epochs 200 --LR 1.00E-07 --balance_ratio 3 --schedule step --momentum 0.9 --expFolder balance_sample-ratio3 --expID 7',
'python main.py --batch_size 2 --optMethod sgd --epochs 200 --LR 1.00E-07 --balance_ratio 3 --schedule step --momentum 0.9 --expFolder balance_sample-ratio3 --expID 8',
'python main.py --batch_size 2 --optMethod rmsprop --epochs 200 --LR 2.00E-07 --balance_ratio 3 --schedule step --momentum 0.9 --expFolder balance_sample-ratio3 --expID 9',
'python main.py --batch_size 2 --optMethod sgd --epochs 200 --LR 2.00E-07 --balance_ratio 3 --schedule step --momentum 0.9 --expFolder balance_sample-ratio3 --expID 10',
'python main.py --batch_size 2 --optMethod rmsprop --epochs 200 --LR 1.00E-07 --balance_ratio 3 --schedule exp --momentum 0 --expFolder balance_sample-ratio3 --expID 11',
'python main.py --batch_size 2 --optMethod adam --epochs 200 --LR 1.00E-07 --balance_ratio 3 --schedule exp --momentum 0 --expFolder balance_sample-ratio3 --expID 12',
'python main.py --batch_size 2 --optMethod sgd --epochs 200 --LR 1.00E-07 --balance_ratio 3 --schedule exp --momentum 0 --expFolder balance_sample-ratio3 --expID 13',
'python main.py --batch_size 2 --optMethod rmsprop --epochs 200 --LR 2.00E-07 --balance_ratio 3 --schedule exp --momentum 0 --expFolder balance_sample-ratio3 --expID 14',
'python main.py --batch_size 2 --optMethod adam --epochs 200 --LR 2.00E-07 --balance_ratio 3 --schedule exp --momentum 0 --expFolder balance_sample-ratio3 --expID 15',
'python main.py --batch_size 2 --optMethod sgd --epochs 200 --LR 2.00E-07 --balance_ratio 3 --schedule exp --momentum 0 --expFolder balance_sample-ratio3 --expID 16',
'python main.py --batch_size 2 --optMethod rmsprop --epochs 200 --LR 1.00E-07 --balance_ratio 3 --schedule exp --momentum 0.9 --expFolder balance_sample-ratio3 --expID 17',
'python main.py --batch_size 2 --optMethod sgd --epochs 200 --LR 1.00E-07 --balance_ratio 3 --schedule exp --momentum 0.9 --expFolder balance_sample-ratio3 --expID 18',
'python main.py --batch_size 2 --optMethod rmsprop --epochs 200 --LR 2.00E-07 --balance_ratio 3 --schedule exp --momentum 0.9 --expFolder balance_sample-ratio3 --expID 19',
'python main.py --batch_size 2 --optMethod sgd --epochs 200 --LR 2.00E-07 --balance_ratio 3 --schedule exp --momentum 0.9 --expFolder balance_sample-ratio3 --expID 20',
'python main.py --batch_size 2 --optMethod rmsprop --epochs 200 --LR 1.00E-07 --balance_ratio 3 --schedule stable --momentum 0 --expFolder balance_sample-ratio3 --expID 21',
'python main.py --batch_size 2 --optMethod adam --epochs 200 --LR 1.00E-07 --balance_ratio 3 --schedule stable --momentum 0 --expFolder balance_sample-ratio3 --expID 22',
'python main.py --batch_size 2 --optMethod sgd --epochs 200 --LR 1.00E-07 --balance_ratio 3 --schedule stable --momentum 0 --expFolder balance_sample-ratio3 --expID 23',
'python main.py --batch_size 2 --optMethod rmsprop --epochs 200 --LR 2.00E-07 --balance_ratio 3 --schedule stable --momentum 0 --expFolder balance_sample-ratio3 --expID 24',
'python main.py --batch_size 2 --optMethod adam --epochs 200 --LR 2.00E-07 --balance_ratio 3 --schedule stable --momentum 0 --expFolder balance_sample-ratio3 --expID 25',
'python main.py --batch_size 2 --optMethod sgd --epochs 200 --LR 2.00E-07 --balance_ratio 3 --schedule stable --momentum 0 --expFolder balance_sample-ratio3 --expID 26',
'python main.py --batch_size 2 --optMethod rmsprop --epochs 200 --LR 1.00E-07 --balance_ratio 3 --schedule stable --momentum 0.9 --expFolder balance_sample-ratio3 --expID 27',
'python main.py --batch_size 2 --optMethod sgd --epochs 200 --LR 1.00E-07 --balance_ratio 3 --schedule stable --momentum 0.9 --expFolder balance_sample-ratio3 --expID 28',
'python main.py --batch_size 2 --optMethod rmsprop --epochs 200 --LR 2.00E-07 --balance_ratio 3 --schedule stable --momentum 0.9 --expFolder balance_sample-ratio3 --expID 29',
'python main.py --batch_size 2 --optMethod sgd --epochs 200 --LR 2.00E-07 --balance_ratio 3 --schedule stable --momentum 0.9 --expFolder balance_sample-ratio3 --expID 30',

]

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)