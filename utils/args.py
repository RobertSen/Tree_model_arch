#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  Author: shen.mouren
"""

import os
import sys
import argparse

_curpath = os.path.dirname(os.path.abspath(__file__))

if _curpath not in sys.path:
    sys.path.append(_curpath)

class Opt(object):
    """
    opt parameters class
    """
    parser = argparse.ArgumentParser(description='task distribution parameters')
    def __init__(self):
        super().__init__()

        self.parser.add_argument('-s', '--seed', default=0, type=int, metavar='SEED',
                                 help='seed for random algorithm', dest='seed')
        self.parser.add_argument('-g', '--gpu', default=0, type=int, metavar='GPU',
                                 help='GPU id to use for training or evalating', dest='gpu')
        self.parser.add_argument('-a', '--arch', default='LR', type=str, metavar='ARCH',
                                 help='architecture backbone', dest='arch')


        self.parser.add_argument('--train_data_dir', default=f'os.path.join(_curpath, "../data/")',
                                 type=str, metavar='TRAIN_DATA_DIR', help='train dataset path', dest='train_data_dir')
        self.parser.add_argument('--eval_data_dir', default=f'os.path.join(_curpath, "../data/")',
                                 type=str, metavar='EVAL_DATA_DIR', help='eval dataset path', dest='eval_data_dir')
        self.parser.add_argument('--fea_mapfile_path', default='', type=str, metavar='FEA_MAPPING_PATH',
                                 help='mapping file path for onehot feature generation', dest='fea_mapfile_path')
        self.parser.add_argument('--shuffle', action='store_true', dest='shuffle')


        self.parser.add_argument('--train_ratio', default=0.8, type=float, metavar='TRAIN_SIZE',
                                 help='train ratio for spliting dataset', dest='train_ratio')
        self.parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='BATCH_SIZE',
                                 help='batch size of training step', dest='batch_size')
        self.parser.add_argument('--output_path', default='output', type=str, metavar='OUTPUT_PATH',
                                 help='output path for store parameter file', dest='output_path')

    @classmethod
    def parse(cls, args_str=''):
        """
        parse parameters from string
        """
        if args_str == '':
            opt = cls.parser.parse_args()
        else:
            args_list = args_str.split()
            opt = cls.parser.parse_args(args_list)
        return opt

if __name__ == '__main__':
    pass
else:
    print(f'import module [{os.path.join(_curpath, __name__)}] succesfully!')
