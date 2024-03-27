# SPDX-FileCopyrightText: 2024 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2024-03-25"
__update__ = "2024-03-25"

import os
from abc import ABC
from typing import List, Dict
import logging
from other_utils import merge_dicts

class AbstractMetricEvaluation(ABC):

    def __init__(self, varname: str, plt_dir: str, model_info: dict, avg_dims: List[str] = ["rlat", "rlon"],
                 eval_dict: Dict = None):
        """
        Abstract class for metric evaluation.
        :param varname: Variable name.
        :param plt_dir: Directory for saving plots.
        :param model_info: Dictionary with keys model_type and model_name
        :param avg_dims: Dimensions for averaging.
        :param eval_dict: Dictionary for configuring evaluation.
        """

        self.required_config_keys = self.required_config_keys()
        self.varname = varname
        self.plt_dir = plt_dir
        self.model_info = model_info
        self.evaluation_dict = eval_dict
        self.avg_dims = avg_dims

        # Create plot directory if it does not exist
        os.makedirs(self.plt_dir, exist_ok=True)

    def __call__(self, *args, **kwargs):
        """
        __call__ method must be implemented in subclass.
        """
        raise NotImplementedError("Method __call__ must be implemented in subclass.")
    
    def get_default_config(self, eval_dict):
        """
        Get default configuration for known variables.
        If the variable for evaluation is unknown, eval_dict cannot be None.
        :param eval_dict: Custom configuration dictionary. Can be None for known variables.
        """
        raise NotImplementedError("Method get_default_config must be implemented in subclass.")
    
    def required_config_keys(self):
        """
        Return required keys for evaluation dictionary.
        """
        raise NotImplementedError("Method required_config_keys must be implemented in subclass.")   
    
    @property
    def evaluation_dict(self):
        return self._evaluation_dict
    
    @evaluation_dict.setter
    def evaluation_dict(self, eval_dict):
        
        default = self.get_default_config(eval_dict)
        default_metrics = list(default.keys())
        
        if not eval_dict:
            eval_dict = default
        else:
            for metric, metric_config in eval_dict.items():
                if metric not in default_metrics:
                    assert all([req_key in metric_config for req_key in self.required_config_keys]), \
                               f"Missing configuration for metric '{metric}'." + \
                               f" Required config keys are: {', '.join(self.required_config_keys)}"
                else:
                    eval_dict[metric] = merge_dicts(default[metric], metric_config, recursive=False)
            
        self._evaluation_dict = eval_dict
