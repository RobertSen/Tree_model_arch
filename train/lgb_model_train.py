#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  Author: shen.mouren
"""
import os
import sys
import time
import traceback
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics

_curpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_curpath)
from base_train import BaseTrain


class LgbModelTrain(BaseTrain):
    """
    lgb model training function
    """
    def __init__(self, arch_model):
        """
        __init__
        """
        super().__init__()
        self.arch_model = arch_model
        self.lgb_params = self.arch_model["model_params"]
        self.num_boost_round = self.arch_model["num_boost_round"]
        self.early_stop_round = self.arch_model["early_stop_round"]
        self.verbose_eval = self.arch_model["verbose_eval"]

    def init(self, opt=None):
        """
        初始化新建目录
        """
        if 'output_name' not in opt or opt.output_name is None:
            self.output_name = time.strftime('%Y%m%d')
        else:
            self.output_name = opt.output_name

        self.output_path = os.path.join(os.path.abspath(opt.output_path), self.output_name)
        os.makedirs(self.output_path, exist_ok=True)
        return True
    
    def train(self, Xtrain, Ytrain):
        """
        [in] train_x, train_y
        [out] return True
        """
        try:
            train_x, dev_x, train_y, dev_y = train_test_split(Xtrain, Ytrain, test_size=0.2, random_state=1)
            train_data = lgb.Dataset(train_x, train_y)
            dev_data = lgb.Dataset(dev_x, dev_y)
            watchlist = [train_data, dev_data]
            self.lgb_model = lgb.train(self.lgb_params, train_data, self.num_boost_round, valid_sets=watchlist, \
                                  early_stopping_rounds=self.early_stop_round, verbose_eval=self.verbose_eval)
            lgb_model_dir = os.path.join(self.output_path, "lgb.model")
            self.lgb_model.save_model(lgb_model_dir)
        except Exception as e:
            print(traceback.format_exc())
            return False
        return True
    
    def val(self, Xeval, Yeval):
        """
        输出混淆矩阵
        """
        predict_y = self.lgb_model.predict(Xeval)
        predict_y_label = [0 if x <= 0.5 else 1 for x in predict_y]
        true_y = Yeval.tolist()
        print(metrics.classification_report(true_y, predict_y_label))
