# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-05-10
# Vesion: 1.0

"""Sub-module with USD-related utilities."""

from __future__ import annotations

import functools
import inspect
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import isaacsim.core.utils.stage as stage_utils
import omni.kit.commands
import omni.log
from isaacsim.core.cloner import Cloner
from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

# from Isaac Sim 4.2 onwards, pxr.Semantics is deprecated
try:
    import Semantics
except ModuleNotFoundError:
    from pxr import Semantics

from isaaclab.utils.string import to_camel_case
from isaaclab.sim import utils
from isaaclab.sim import schemas
from isaaclab.sim.utils import apply_nested

"""
Attribute - Setters.
"""

def safe_set_attribute_on_usd_prim(prim: Usd.Prim, attr_name: str, value: Any, camel_case: bool):
    """Set the value of a attribute on its USD prim.

    This function is a wrapper around the isaac lab funtion  `safe_set_attribute_on_usd_prim`_.

    The function in isaac lab not support String, so if value is string, set attribute in this function, if value is not string, use funtion in isaac lab.

    """
    # if value is None, do nothing
    if value is None:
        return
    # convert attribute name to camel case
    if camel_case:
        attr_name = to_camel_case(attr_name, to="cC")
    # resolve sdf type based on value
    if isinstance(value, str):
        sdf_type = Sdf.ValueTypeNames.Bool
        # change property
        omni.kit.commands.execute(
            "ChangePropertyCommand",
            prop_path=Sdf.Path(f"{prim.GetPath()}.{attr_name}"),
            value=value,
            prev=None,
            type_to_create_if_not_exist=sdf_type,
            usd_context_name=prim.GetStage(),
        )

    else:
        utils.safe_set_attribute_on_usd_prim(prim, attr_name, value, camel_case)

# @apply_nested
# def set_rigid_body_enbale(
#     prim_path: str | Sdf.Path,
#     enable:bool,
#     stage: Usd.Stage | None = None,
# ):
#     """Set `rigidBodyEnabled` attribute of rigid body."""
#     success = False
#     # resolve stage
#     if stage is None:
#         stage = stage_utils.get_current_stage()
#     # check if prim and material exists
#     if not stage.GetPrimAtPath(prim_path).IsValid():
#         raise ValueError(f"Target prim '{prim_path}' does not exist.")
#     # get USD prim
#     prim: Usd.Prim = stage.GetPrimAtPath(prim_path)
#     # check if prim has 'physics:rigidBodyEnabled'
#     if prim.GetAttribute("physxRigidBody:disableGravity").IsValid():
#         safe_set_attribute_on_usd_prim(prim,"physxRigidBody:disableGravity",enable,camel_case=False)
#         success = True
#     return success