# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-05-10
# Vesion: 1.0

""" Python Modules  """ 
from dataclasses import MISSING
from typing import Literal

""" IsaacLab Modules  """ 
from isaaclab.utils.configclass import configclass

""" PsiLab Modules  """ 
from psilab.random.material_random_cfg import MaterialRandomCfg

@configclass
class TaskRandomCfg():

    enable : bool = MISSING  # type: ignore
    """Whether task radndom is enabled or disenable."""

    target_list : list[str] = MISSING # type: ignore
    """Rigid/Deformable objects list of which env will pick one or some as target."""

    target_num : int = MISSING # type: ignore
    """Target number."""

    target_indexs : list[int] = None # type: ignore
    """The indexs of target in target list."""

    obstacle_list : list[str] = MISSING # type: ignore
    """Rigid/Deformable objects list of which env will pick one or some as obstacle."""

    obstacle_num : int = MISSING # type: ignore
    """Obstacle number."""

    obstacle_indexs : list[int] = None # type: ignore
    """The indexs of target in obstacle list."""

    select_obstacle_from_target : bool  = MISSING # type: ignore
    """Whether select obsacle from obstacle_list or target_list."""

    position_offset : list[float,float,float] = None # type: ignore
    """The offset of center position which disactivate objects will be placed around."""

    space : float = None # type: ignore
    """The space between two disactivate objects."""