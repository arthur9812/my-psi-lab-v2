# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from __future__ import annotations
from dataclasses import MISSING

""" Common Modules  """ 
import time
import torch
import numpy
import random
import warnings
from collections import deque
import matplotlib.pyplot as plt
import os
import wandb
from datetime import datetime

import rl_games.common.a2c_common 
global fps_step

""" Isaac Sim Modules  """ 
import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
from isaacsim.core.utils.prims import get_prim_at_path
# # from Isaac Sim 4.2 onwards, pxr.Semantics is deprecated
# try:
#     import Semantics
# except ModuleNotFoundError:
#     from pxr import Semantics


""" Isaac Lab Modules  """ 
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg,PhysxCfg,RenderCfg
from isaaclab.utils import configclass
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
)
from isaaclab.envs import DirectRLEnvCfg, DirectRLEnv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import (  # isort:skip
    quat_from_euler_xyz
)
from isaaclab.envs.common import SpaceType, ViewerCfg


""" Psi Lab Modules  """
from psilab.envs.rp_env import RPEnv 
from psilab.envs.rp_env_cfg import RPEnvCfg
# from psila.assets.realman_inspire_no_camera import REALMAN_INSPIRE_NO_CAMERA_CFG
# from psi_rl import PSI_RL_USD_ASSET_DIR
from psilab.utils.wandb_utils import WandbLog

from psilab.configs.device.vuer_psi_dc_01 import VUER_PSI_DC_01_CFG
from psilab import OUTPUT_DIR

@configclass
class ReplayEnvCfg(RPEnvCfg):
    """Configuration for Rl environment."""

    # fake params 
    episode_length_s = 1 * 210 / 60.0
    decimation = 2
    action_scale = 0.5
    action_space = 13
    observation_space = 130
    state_space = 130



    # viewer config
    viewer = ViewerCfg(
        eye=(2.2,0.0,1.2),
        lookat=(-15.0,0.0,0.3)
    )

    # simulation  config
    sim: SimulationCfg = SimulationCfg(
        dt = 1 / 120, 
        render_interval=decimation,
        physx = PhysxCfg(
            solver_type = 1, # 0: pgs, 1: tgs
            max_position_iteration_count = 32,
            max_velocity_iteration_count = 32,
            bounce_threshold_velocity = 0.002,
            # enable_ccd=False,
            gpu_found_lost_pairs_capacity = 137401003
        ),
        render=RenderCfg(),

    )

    # scene config
    scene = MISSING  # type: ignore



class ReplayEnv(RPEnv):

    cfg: ReplayEnvCfg

    def __init__(self, cfg: ReplayEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

       