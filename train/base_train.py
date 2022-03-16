#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  Author: shen.mouren
"""

"""
训练过程基类
"""

import os
import sys
import abc

_curpath = os.path.dirname(os.path.abspath(__file__))

class BaseTrain(object):
    """
    训练基类
    """
    @abc.abstractmethod
    def init(self, opt):
        """
        init model context
        """
        return True

    @abc.abstractmethod
    def train(self, Xtrain, Ytrain):
        """
        trainning process
        """
        return True

    @abc.abstractmethod
    def val(self, Xteset, Ytest):
        """
        validation process
        """
        return True

if __name__ == '__main__':
    pass
else:
    print(f'import module [{os.path.join(_curpath, __name__)}] successfully!')

