# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from dataclasses import MISSING

""" IsaacLab Modules  """ 
from isaaclab.utils.configclass import configclass


@configclass
class RigidRandomCfg():
    """Configuration for rigid random options."""

    fake_random : bool =  MISSING # type: ignore
    """Configuration for light random options.

    - ``True``: parameters are random within the given range.
    - ``False``: parameters are random selected from the given list.
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
    
    # materials : None | list[list[float,float,float,float]] =  MISSING # type: ignore

    