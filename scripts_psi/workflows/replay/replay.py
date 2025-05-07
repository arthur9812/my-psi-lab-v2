# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Arguments parse """
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates lego grasp task demo from gym.")

parser.add_argument("--task", type=str, default="", help="Name of the task.")
parser.add_argument("--hdf5_file", type=str, default="", help="Name of the task.")
parser.add_argument("--json_file", type=str, default="", help="Name of the task.")



""" Must First Start APP, or import omni.isaac.lab.sim as sim_utils will be error."""
from isaaclab.app import AppLauncher

# append AppLauncher cli args 
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# store args befor create app as it will pop some arg from args_cli
enable_cameras = args_cli.enable_cameras

# launch omniverse app
app_launcher = AppLauncher(args_cli)


""" Common Modules  """ 
import os
import sys
import math
import torch
import time
import json
import gymnasium as gym
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner
from datetime import datetime

""" Isaac Lab Modules  """ 
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg,load_cfg_from_registry
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
print('isaaclab_tasks.direct' in sys.modules)
from psilab.envs.tp_env import TPEnv

""" Psi RL Modules  """ 
# import psilab.tasks # noqa: F401
import psilab_tasks
from psilab.scene.sence_cfg import SceneCfg
from psilab.envs.rp_env import RPEnv
from psilab.envs.rp_env_cfg import RPEnvCfg

from psilab.utils.config_utils import scene_cfg
from psilab_tasks.utils import parse_scene_cfg,parse_rp_env_cfg
# parse argumanets for isaac lab rl env config
# env_cfg= parse_env_cfg(
#     args_cli.task, 
#     device=args_cli.device,
#     num_envs=1,
#     # use_fabric=not args_cli.disable_fabric
#     )

env_cfg = RPEnvCfg(
    decimation=1,
    episode_length_s=1,
    observation_space=1,
    action_space=1,
    output_folder=None
)
# parse argumanets for psi lab rl env config
env_cfg = parse_rp_env_cfg(
    env_cfg,
    args_cli.hdf5_file,
    args_cli.json_file
)

# parse argumanets for psi lab scene config
env_cfg.scene = parse_scene_cfg(
    args_cli.task, 
    True,
    args_cli.json_file,
    1,
)

# clear camera configs in scene while "enable_cameras" flag is True
if enable_cameras is False:
    env_cfg.scene.cameras_cfg ={}
    env_cfg.scene.tiled_cameras_cfg = {}
    for robot_cfg in env_cfg.scene.robots_cfg.values():
        robot_cfg.cameras = {} # type: ignore
        robot_cfg.tiled_cameras = {} # type: ignore

# create env
env = gym.make(args_cli.task, cfg=env_cfg)

# reset env before loop
env.reset()

while(True):
    #env step
    env.step(torch.zeros(1))
