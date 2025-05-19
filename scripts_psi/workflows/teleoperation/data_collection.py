# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Arguments parse """
import argparse

# add argparse arguments from Psi
parser = argparse.ArgumentParser(description="This script demonstrates lego grasp task demo from gym.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--enable_wandb", action="store_true", default=False, help="Whether update data to wandb or not.")
parser.add_argument("--enable_json", action="store_true", default=False, help="Whether create scene from json or not.")
parser.add_argument("--scene", type=str, default=None, help="Scene.")
parser.add_argument("--json_file", type=str, default=None, help="Scene json file.")
parser.add_argument("--enable_output", action="store_true", default=False, help="Whether output data to hdf5 files or not.")
parser.add_argument("--output_folder", type=str, default=None, help="Hdf5 files folder.")
parser.add_argument("--sample_step", type=int, default=1, help="Simulation steps per sample step") 


""" Must First Start APP, or import omni.isaac.lab.sim as sim_utils will be error."""
from isaaclab.app import AppLauncher

# append AppLauncher cli args 
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)


""" Common Modules  """ 
import os
import sys
import math
import torch
import json
import gymnasium as gym
import importlib
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
from psilab.envs.tp_env import TPEnv
from psilab_tasks.utils import parse_scene_cfg,parse_rl_env_cfg

# parse argumanets for isaac lab rl env config
env_cfg= parse_env_cfg(
    args_cli.task, 
    device=args_cli.device,
    num_envs=1,
    # use_fabric=not args_cli.disable_fabric
    )

# parse argumanets for psi lab rl env config
env_cfg = parse_rl_env_cfg(
    env_cfg,
    args_cli.seed,
    args_cli.enable_wandb,
    args_cli.enable_output,
    args_cli.output_folder,
    args_cli.sample_step
)

# parse argumanets for psi lab scene config
env_cfg.scene = parse_scene_cfg(
    args_cli.task, 
    args_cli.enable_json,
    args_cli.scene,
    args_cli.json_file,
    args_cli.num_envs,
)

# create env
env = gym.make(args_cli.task, cfg=env_cfg)

# reset env before loop
env.reset()

while(True):
    # env step
    env.step(torch.zeros(1))
