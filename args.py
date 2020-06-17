import argparse

parser = argparse.ArgumentParser(description='args for project xxx')

# pre-trained model name
parser.add_argument('--pretrain_model', default='resnext101_32x8d', type=str,
                    choices=['resnext101_32x8d', 'resnext101_32x16d'],
                    help='model_name selected in train')


parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--category', default='Mnist', type=str)


# Directory
parser.add_argument('--latest_ckpt', default="", type=str, help='path to latest checkpoint')
parser.add_argument('--ckpt_dir', default="", type=str, help='path to save checkpoint')


args = parser.parse_args()