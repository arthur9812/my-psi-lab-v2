# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from dataclasses import MISSING
from typing import Literal

""" IsaacLab Modules  """ 
from isaaclab.utils.configclass import configclass

""" PsiLab Modules  """ 
from psilab.random.material_random_cfg import MaterialRandomCfg

@configclass
class RigidRandomCfg():
    """Configuration for rigid random options."""

    random_type : Literal["range", "list"]  = MISSING # type: ignore
    """Configuration for light random options.

    - ``range``: parameters are random within the given range.
    - ``list``: parameters are random selected from the given list.
    """

    random_position : bool =  MISSING # type: ignore
    """Whether the position of rigid is random."""

    random_orientation : bool =  MISSING # type: ignore
    """Whether the orientation of rigid is random."""

    random_material : bool =  MISSING # type: ignore
    """Whether the material of rigid is random."""

    position_range : None | list[float,float,float] =  None # type: ignore
    """The delta position range which is [delat_x_max,delat_y_max,delat_z_max].
    """

    position_list : None | list[list[float,float,float]] =  None # type: ignore
    """The delta position list which is [delat_x,delat_y,delat_z],...]."""

    orientation_list : None | list[list[float,float,float,float]] =  None # type: ignore
    """The orientation list which is [w,x,y,z],...]."""
    
    material_cfg : None | MaterialRandomCfg =  None # type: ignore
    """The material random config."""

    