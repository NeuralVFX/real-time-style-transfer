#!/usr/bin/env python
import argparse
from style_transfer import StyleTransfer


parser = argparse.ArgumentParser()

parser.add_argument("cmd", help=argparse.SUPPRESS, nargs="*")
parser.add_argument('--dataset', nargs='?', default='VOCdevkit/VOC2007/JPEGImages/', type=str)
parser.add_argument('--in_channels', nargs='?', default=3, type=int)
parser.add_argument('--batch_size', nargs='?', default=4, type=int)
parser.add_argument('--gen_filters', nargs='?', default=128, type=int)
parser.add_argument('--gen_layers', nargs='?', default=2, type=int)
parser.add_argument('--res_blocks', nargs='?', default=5, type=int)
parser.add_argument('--vgg_layers_c', nargs='?', default=2, type=int)
parser.add_argument('--vgg_layers_s', type=int, nargs='+', default=[3, 8, 15, 22])
parser.add_argument('--style_weight', nargs='?', default=10000000000, type=int)
parser.add_argument('--content_weight', nargs='?', default=10000, type=int)
parser.add_argument('--res', nargs='?', default=256, type=int)
parser.add_argument('--lr', nargs='?', default=1e-3, type=float)
parser.add_argument('--train_epoch', nargs='?', default=60, type=int)
parser.add_argument('--test_perc', nargs='?', default=.5, type=float)
parser.add_argument('--data_perc', nargs='?', default=.5, type=float)
parser.add_argument('--beta1', nargs='?', default=.5, type=float)
parser.add_argument('--beta2', nargs='?', default=.999, type=float)
parser.add_argument('--workers', nargs='?', default=4, type=int)
parser.add_argument('--save_every', nargs='?', default=5, type=int)
parser.add_argument('--ids', type=int, nargs='+', default=[10, 20])
parser.add_argument('--style_image', nargs='?', default='franc.jpg', type=str)
parser.add_argument('--save_root', nargs='?', default='franc_style', type=str)
parser.add_argument('--load_state', nargs='?', type=str)

params = vars(parser.parse_args())


# if load_state arg is not used, then train model from scratch
if __name__ == '__main__':
    style_transfer = StyleTransfer(params)
    if params['load_state']:
        style_transfer.load_state(params['load_state'])
    else:
        print('Starting From Scratch')
    style_transfer.train()
