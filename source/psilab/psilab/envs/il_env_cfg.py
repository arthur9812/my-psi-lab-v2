# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from dataclasses import MISSING

""" IsaacLab Modules  """ 
from isaaclab.utils import configclass

""" PsiLab Modules  """ 
from psilab.envs.rl_env_cfg import RLEnvCfg

@configclass
class ILEnvCfg(RLEnvCfg):
    """Configuration for an imitation learning environment.
    """
    
    pass



