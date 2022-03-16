#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  Author: shen.mouren
"""

import os
import sys
import time
import joblib
import traceback
import lightgbm as lgb
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

_curpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_curpath)
from base_train import BaseTrain


class StackModelTrain(BaseTrain):
    """
    model training function
    xgboost, lightgbm, ridge stacking
    """
    def __init__(self, arch_model, k_flods=3):
        super().__init__()
        self.arch_model = arch_model
        self.arch_model_xgb = self.arch_model["xgboost"]
        self.arch_model_lgb = self.arch_model["lightgbm"]

        self.xgb_params = self.arch_model_xgb["model_params"]
        self.xgb_num_boost_round = self.arch_model_xgb["num_boost_round"]
        self.xgb_early_stop_round = self.arch_model_xgb["early_stop_round"]
        self.xgb_verbose_eval = self.arch_model_xgb["verbose_eval"]

        self.lgb_params = self.arch_model_lgb["model_params"]
        self.lgb_num_boost_round = self.arch_model_lgb["num_boost_round"]
        self.lgb_early_stop_round = self.arch_model_lgb["early_stop_round"]
        self.lgb_verbose_eval = self.arch_model_lgb["verbose_eval"]

        self.k_flods = k_flods

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
        each_models = self.__dict__
        try:
            train_x = Xtrain
            train_y = Ytrain
            #交叉验证的折数,超参是可以改的
            folds = KFold(n_splits=self.k_flods, shuffle=True)
            #xgboost训练与lightgbm训练
            oof_xgb = np.zeros(len(train_x))
            oof_lgb = np.zeros(len(train_x))
            for flod_nums, (train_idx, val_idx) in enumerate(folds.split(train_x, train_y)):
                #交叉验证中K-1折的部分
                cur_dev_x = train_x.iloc[train_idx]
                cur_dev_y = train_y.iloc[train_idx]
                #交叉验证中1折的部分
                cur_val_x = train_x.iloc[val_idx]
                cur_val_y = train_y.iloc[val_idx]
                #训练过程
                #xgboost
                xgb_train_data = xgb.DMatrix(cur_dev_x, cur_dev_y)
                xgb_dev_data = xgb.DMatrix(cur_val_x, cur_val_y)
                xgb_watchlist = [(xgb_train_data, 'train'), (xgb_dev_data, 'valid_data')]
                xgb_train_model = xgb.train(dtrain=xgb_train_data, num_boost_round=self.xgb_num_boost_round, \
                                            evals=xgb_watchlist, early_stopping_rounds=self.xgb_early_stop_round, \
                                            verbose_eval=self.xgb_verbose_eval, params=self.xgb_params)

                oof_xgb[val_idx] = xgb_train_model.predict(xgb.DMatrix(cur_val_x))
                xgb_model_name = "xgb_{}.model".format(str(flod_nums))
                xgb_model_dir = os.path.join(self.output_path, xgb_model_name)
                xgb_train_model.save_model(xgb_model_dir)
                each_models['xgb_model_' + str(flod_nums)] = xgb_train_model
                #lightgbm
                lgb_train_data = lgb.Dataset(cur_dev_x, cur_dev_y)
                lgb_dev_data = lgb.Dataset(cur_val_x, cur_val_y)
                lgb_watchlist = [lgb_train_data, lgb_dev_data]
                lgb_train_model = lgb.train(self.lgb_params, lgb_train_data, self.lgb_num_boost_round, \
                                            valid_sets=lgb_watchlist, early_stopping_rounds=self.lgb_early_stop_round, \
                                            verbose_eval=self.lgb_verbose_eval)

                oof_lgb[val_idx] = lgb_train_model.predict(cur_val_x)
                lgb_model_name = "lgb_{}.model".format(str(flod_nums))
                lgb_model_dir = os.path.join(self.output_path, lgb_model_name)
                lgb_train_model.save_model(lgb_model_dir)
                each_models['lgb_model_' + str(flod_nums)] = lgb_train_model
            
            #stacking
            train_stack = np.vstack([oof_xgb, oof_lgb]).transpose()

            for flod_nums, (train_idx, val_idx) in enumerate(folds.split(train_stack, train_y)):
                lr_train_x, lr_train_y_np = train_stack[train_idx], train_y.iloc[train_idx].values
                lr_val_x, lr_val_y_np = train_stack[val_idx], train_y.iloc[val_idx].values
                lr_train_y = lr_train_y_np.reshape(1, -1)[0]
                lr_model = LogisticRegression()
                lr_model.fit(lr_train_x, lr_train_y)
                each_models['lr_model_' + str(flod_nums)] = lr_model
                lr_model_name = "lr_{}.model".format(str(flod_nums))
                lr_model_dir = os.path.join(self.output_path, lr_model_name)
                #sklearn的模型，最好是用joblib保存
                joblib.dump(lr_model, lr_model_dir)
        except Exception as e:
            print(traceback.format_exc())
            return False
        return True
    
    def val(self, Xeval, Yeval):
        """
        输出混淆矩阵
        """
        true_y = Yeval.tolist()
        predict_x = Xeval
        predict_xgb = np.zeros(len(predict_x))
        predict_lgb = np.zeros(len(predict_x))
        predict_lr = np.zeros(len(predict_x))
        for i in range(self.k_flods):
            exec('predict_xgb += self.xgb_model_{}.predict(xgb.DMatrix(predict_x)) / self.k_flods'.format(i))
            exec('predict_lgb += self.lgb_model_{}.predict(predict_x) / self.k_flods'.format(i))
        predict_stack = np.vstack([predict_xgb, predict_lgb]).transpose()
        for i in range(self.k_flods):
            exec('predict_lr += self.lr_model_{}.predict(predict_stack) / self.k_flods'.format(i))
        predict_lr_label = [0 if x <= 0.5 else 1 for x in predict_lr]
        print(metrics.classification_report(true_y, predict_lr_label))
