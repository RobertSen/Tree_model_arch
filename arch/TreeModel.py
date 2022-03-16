#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  Author: shen.mouren
"""
import os
import sys
import json

_curpath = os.path.dirname(os.path.abspath(__file__))
if _curpath not in sys.path:
    sys.path.append(_curpath)
import base_arch


class TreeModel(base_arch.BaseArch):
    """
    根据参数调用不同的文件
    """
    def __init__(self, archname):
        """
        init
        """
        self.arch_name = archname
        self.model_config = os.path.join(_curpath, "arch_config/{}_params.json".format(self.arch_name))
        if not os.path.exists(self.model_config):
            raise "{} model config is not exits".format(self.arch_name)

    def init_from_json(self, config_file):
        """
        read model from json
        [in] config_file: json
        """
        with open(config_file, "r") as fr:
            return json.loads(fr.read())

    def __model__(self):
        """
        __model__
        """
        self.model_json = self.init_from_json(self.model_config)
        return self.model_json

    def arch(self):
        """
        arch
        """
        return self.__model__()


if __name__ == "__main__":
    pass
