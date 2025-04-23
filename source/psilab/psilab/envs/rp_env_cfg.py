# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 

from dataclasses import MISSING

""" Isaac Lab Modules  """ 
from isaaclab.utils import configclass

""" Psi Lab Modules  """ 
from psilab.envs.rl_env_cfg import RLEnvCfg

@configclass
class RPEnvCfg(RLEnvCfg):
    
    h5_file : str = MISSING # type: ignore
    """ H5 file to replay. """



