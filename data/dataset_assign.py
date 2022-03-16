#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  Author: shen.mouren
"""
import pandas as pd

mode_dict = {
    'train': "deal_data",
    'eval': "deal_data",
    'infer': "deal_predict_data"
}

class DatasetAssign(object):
    """
    """
    def __init__(self, data_dir, mode='train'):
        """
        __init__
        """
        self.data_file = data_dir
        if mode not in mode_dict:
            raise "mode type error"
        self.mode = mode
        self.data_frame = pd.read_csv(self.data_file, index_col=0, header=0)

    def data_transform(self):
        """
        process of data
        """
        x_data = self.data_frame.iloc[:, :-1]
        return x_data
    
    def data_predict_transform(self):
        """
        predict process of data
        """
        x_data = self.data_frame
        return x_data

    def target_transform(self):
        """
        process of target data
        """
        y_data = self.data_frame.iloc[:, -1]
        return y_data
    
    def deal_data(self):
        """
        deal data
        feature data & target data
        """
        feature_data = self.data_transform()
        target_data = self.target_transform()
        return feature_data, target_data
    
    def deal_predict_data(self):
        """
        feature data
        """
        feature_data = self.data_predict_transform()
        return feature_data
    
    def get_data(self):
        """
        根绝mode读取数据
        """
        return getattr(self, mode_dict[self.mode])()

if __name__ == "__main__":
    pass
