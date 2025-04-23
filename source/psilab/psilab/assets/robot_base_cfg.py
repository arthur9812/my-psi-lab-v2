# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0


""" IsaacLab Modules  """ 
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.assets.articulation import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors.camera import CameraCfg,Camera


""" PsiLab Modules  """ 
from psilab.assets.robot_base import RobotBase
from psilab.controllers.differential_ik_cfg import DiffIKControllerCfg

@configclass
class RobotBaseCfg(ArticulationCfg):
    """Configuration parameters for an robot_base with cameras and ik controllers.""" 

    ##
    # Initialize configurations.
    ##
    class_type: type = RobotBase

    init_state: ArticulationCfg.InitialStateCfg = ArticulationCfg.InitialStateCfg()
    """Initial state of the articulated object. Defaults to identity pose with zero velocity and zero joint state."""

    diff_ik_controllers: dict[str, DiffIKControllerCfg] = MISSING   # type: ignore
    """Differential IK Controllers Config for the robot with corresponding joint group names."""

    cameras: dict[str, CameraCfg] = MISSING     # type: ignore
    """Cameras Config for the robot with corresponding camera names."""




