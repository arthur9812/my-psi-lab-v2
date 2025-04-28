# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from dataclasses import MISSING
from typing import Literal

""" IsaacLab Modules  """ 
from isaaclab.utils.configclass import configclass


@configclass
class LightRandomCfg():
    """Configuration for light random options."""

    fake_random : bool =  MISSING # type: ignore
    """Configuration for light random options.

    - ``True``: parameters are random within the given range.
    - ``False``: parameters are random selected from the given list.
    """

    random_intensity : bool =  MISSING # type: ignore
    """Whether the intensity of light is random."""

    random_color : bool =  MISSING # type: ignore
    """Whether the color of light is random."""

    intensity_range : None | list[float,float] =  MISSING # type: ignore
    """The intensity range which is [lower_limit, upper_limit]."""

    color_range : None | list[list[float,float]] =  MISSING # type: ignore
    """The color range.
    - Notes: \n
    [[red_lower_limit, red_upper_limit],[green_lower_limit, green_upper_limit],[blue_lower_limit, blue_upper_limit],]
    """

    intensity_list : None | list[float] =  MISSING # type: ignore
    """The intensity list."""

    color_list : None | list[float,float,float] =  MISSING # type: ignore
    """The color list(RGB)."""


    