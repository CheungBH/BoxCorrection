import argparse

parser = argparse.ArgumentParser(description='PyTorch AlphaPose Training')

"----------------------------- General options -----------------------------"
parser.add_argument('--expID', default='default', type=str,
                    help='Experiment ID')
parser.add_argument('--expFolder', default='test', type=str,
                    help='Experiment folder')

"----------------------------- Model options -----------------------------"
parser.add_argument('--loadModel', default=None, type=str,
                    help='Provide full path to a previously trained model')
parser.add_argument('--resume', action='store_true',
                    help='resume training')

"----------------------------- Hyperparameter options -----------------------------"

parser.add_argument('--LR', default=1e-3, type=float,
                    help='Learning rate')
parser.add_argument('--optMethod', default='rmsprop', type=str,
                    help='Optimization method: rmsprop | sgd | nag | adadelta')

"----------------------------- Training options -----------------------------"
parser.add_argument('--nEpochs', default=120, type=int,
                    help='Number of hourglasses to stack')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='Current epoch')
parser.add_argument('--batch-size', default=2, type=int,
                    help='Train-batch size')
parser.add_argument('--trainIters', default=0, type=int,
                    help='Total train iters')
parser.add_argument('--valIters', default=0, type=int,
                    help='Total valid iters')
parser.add_argument('--train_worker', default=5, type=int,
                    help='num worker of train')
parser.add_argument('--val_worker', default=2, type=int,
                    help='num worker of val')
parser.add_argument('--save_interval', default=1, type=int,
                    help='interval')

opt, _ = parser.parse_known_args()

