import argparse

parser = argparse.ArgumentParser(description='PyTorch AlphaPose Training')

"----------------------------- General options -----------------------------"
parser.add_argument('--expID', default='default', type=str,
                    help='Experiment ID')
parser.add_argument('--expFolder', default='test', type=str,
                    help='Experiment folder')

"----------------------------- Data options ------------------------------"
parser.add_argument('--dataset_root', default='h5', type=str,
                    help='Experiment ID')
parser.add_argument('--balance_ratio', default=0, type=int, help="The ratio of neg: pos sample")

"----------------------------- Model options -----------------------------"
parser.add_argument('--loadModel', default=None, type=str,
                    help='Provide full path to a previously trained model')
parser.add_argument('--resume', action='store_true',
                    help='resume training')

"----------------------------- Hyperparameter options -----------------------------"

parser.add_argument('--LR', default=2E-7, type=float,
                    help='Learning rate')
parser.add_argument('--optMethod', default='adam', type=str,
                    help='Optimization method: rmsprop | sgd | nag | adadelta')

"----------------------------- Training options -----------------------------"
parser.add_argument('--epochs', default=120, type=int,
                    help='Number of hourglasses to stack')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='Current epoch')
parser.add_argument('--batch_size', default=2, type=int,
                    help='Train-batch size')
parser.add_argument('--iterations', default=0, type=int,
                    help='Total train iters')
parser.add_argument('--num_worker', default=5, type=int,
                    help='num worker of train')
parser.add_argument('--save_interval', default=2, type=int,
                    help='interval')
parser.add_argument('--device', default="cuda", type=str,
                    help='interval')

opt, _ = parser.parse_known_args()

