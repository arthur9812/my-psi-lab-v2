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
from psilab.devices.vuer_tp_cfg import VuerTpCfg

@configclass
class TPEnvCfg(RLEnvCfg):
    """Configuration for an tele operation environment.
    """

    device_cfg : VuerTpCfg = MISSING  # type: ignore
    """ Vuer device config. """

    ouput_folder: str = MISSING  # type: ignore
    """ Data Ouptut Folder. """

    sample_step : int = MISSING  # type: ignore
    """ Agent step numbers per Sample step. """
