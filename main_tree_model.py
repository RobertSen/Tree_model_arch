#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  Author: shen.mouren
"""
import sys

from arch import TreeModel
from data import dataset_assign
from train import xgb_model_train
from train import lgb_model_train
from train import stack_tree_model_train
from utils import args

def main():
    param_str = '--arch xgboost \
                 --train_data_dir data/train_data/tree_model_train.csv \
                 --eval_data_dir data/train_data/tree_model_eval.csv \
                 --output_path output \
                 '
    #param_str = '--arch lightgbm \
    #             --train_data_dir data/train_data/tree_model_train.csv \
    #             --eval_data_dir data/train_data/tree_model_eval.csv \
    #             --output_path output \
    #             '
    #param_str = '--arch stacking \
    #             --train_data_dir data/train_data/tree_model_train.csv \
    #             --eval_data_dir data/train_data/tree_model_eval.csv \
    #             --output_path output \
    #             '
    #参数加载成功
    opt = args.Opt().parse(param_str)
    #模型参数加载成功
    arch_xgb = TreeModel.TreeModel(opt.arch).arch()
    #arch_lgb = TreeModel.TreeModel(opt.arch).arch()
    #arch_stacking = TreeModel.TreeModel(opt.arch).arch()
    #加载数据
    train_dataset = dataset_assign.DatasetAssign(data_dir=opt.train_data_dir,
                                                mode="train")
    Xtrain, Ytrain = train_dataset.get_data()

    #加载数据后需要初始化训练类
    trainer = xgb_model_train.XgbModelTrain(arch_xgb)
    #trainer = lgb_model_train.LgbModelTrain(arch_lgb)
    #trainer = stack_tree_model_train.StackModelTrain(arch_stacking)

    if not trainer.init(opt):
        print(f'trainer init failed!')
        return False
    if not trainer.train(Xtrain, Ytrain):
        print(f'trainer training process failed!')
        return False

    eval_dataset = dataset_assign.DatasetAssign(data_dir=opt.eval_data_dir,
                                                mode="eval")
    Xeval, Yeval = eval_dataset.get_data()
    trainer.val(Xeval, Yeval)


if __name__ == "__main__":
    main()

