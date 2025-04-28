# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-28
# Vesion: 1.0

""" Python Modules  """ 
from dataclasses import MISSING
from typing import Literal

""" Common Modules"""
import numpy

""" IsaacLab Modules  """ 
from isaaclab.utils.configclass import configclass


@configclass
class MaterialRandomCfg():
    """Configuration for material random options(Only for MDL Material)."""

    enable_random : bool = False
    """Whether the material is random."""

    random_type : Literal["range", "list"]  = MISSING # type: ignore
    """Configuration for light random options.

    - ``range``: parameters are random within the given range.
    - ``list``: parameters are random selected from the given list.
    """

    material_type : Literal["color", "texture", "colored_texture"]  = MISSING # type: ignore
    """Configuration for material type.
    - ``color``: the material is show as rgb color.
    - ``texture``: the material is show as texture map.
    - ``colored_texture``: the material is show as texture map with rgb color.
    """

    color_range : list[list[numpy.uint8,numpy.uint8,numpy.uint8]] =  MISSING # type: ignore
    """The RGB range.
    - ``color_range[0]``: The minimum value of the RGB color range.
    - ``color_range[1]``: The maximum value of the RGB color range.
    """

    color_list : list[numpy.uint8,numpy.uint8,numpy.uint8] = MISSING # type: ignore
    """The RGB list."""

    texture_list : list[str] = MISSING # type: ignore
    """The texture list."""

    shader_path : str = MISSING # type: ignore
    """The Shader path of object material in isaac."""

    