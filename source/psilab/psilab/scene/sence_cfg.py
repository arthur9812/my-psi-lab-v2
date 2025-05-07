# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from dataclasses import MISSING

""" IsaacLab Modules  """ 
from isaaclab.utils.configclass import configclass
from isaaclab.scene.interactive_scene_cfg import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.camera import CameraCfg,TiledCameraCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.assets import (
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
)

""" PsiLab Modules  """ 
from psilab.assets.robot_base_cfg import RobotBaseCfg
from psilab.random.random_cfg import RandomCfg


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for scene."""

    global_light_cfg: AssetBaseCfg = MISSING # type: ignore
    """The global light configuration."""

    local_lights_cfg : dict[str, AssetBaseCfg] = MISSING # type: ignore
    """The local lights configuration dict."""

    robots_cfg : dict[str, RobotBaseCfg] = MISSING # type: ignore
    """The robots configuration dict."""

    static_objects_cfg : dict[str, AssetBaseCfg] = MISSING # type: ignore
    """The static objects configuration dict."""

    rigid_objects_cfg : dict[str, RigidObjectCfg] = MISSING # type: ignore
    """The rigid objects configuration dict."""

    deformable_objects_cfg : dict[str, DeformableObjectCfg] = MISSING # type: ignore
    """The deformable objects configuration dict."""

    cameras_cfg: dict[str, CameraCfg] = MISSING  # type: ignore
    """The cameras configuration dict."""

    tiled_cameras_cfg: dict[str, TiledCameraCfg] = MISSING  # type: ignore
    """The tiled cameras configuration dict."""

    contact_sensors_cfg: dict[str, ContactSensorCfg] = MISSING  # type: ignore
    """The contact sensors configuration dict."""

    marker_cfg : None | VisualizationMarkersCfg = MISSING  # type: ignore
    """The debug markers configuration."""

    random: None | RandomCfg = MISSING   # type: ignore
    """The random configuration."""
