#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  Author: shen.mouren
"""

"""
网络框架基类
"""

import os
import sys
import abc

_curpath = os.path.dirname(os.path.abspath(__file__))

class BaseArch(object):
    """
    网络框架基类
    """
    @abc.abstractmethod
    def arch(self):
        """
        architecture
        """
        return True
