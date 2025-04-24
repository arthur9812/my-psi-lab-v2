# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Arguments parse """
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates lego grasp task demo from gym.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="", help="Name of the task.")

parser.add_argument("--output_dir", type=str, default=None, help="Path to model checkpoint.")

# add argparse arguments from Psi
parser.add_argument("--enable_wandb", action="store_true", default=False, help="Whether update Data to Wandb or not.")
parser.add_argument("--enable_json", action="store_true", default=False, help="Create Scene from json.")
parser.add_argument("--scene_file", type=str, default=None, help="Scene json file path.")
parser.add_argument("--enable_store", action="store_true", default=False, help="Whether Store Data to files or not.")


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
from psilab.utils.config_utils import scene_cfg

# from psilab.envs.tp_env import TPEnvDataCollectWrapper
# create env
env_cfg= parse_env_cfg(
    args_cli.task, 
    device=args_cli.device,
    num_envs=1,
    # use_fabric=not args_cli.disable_fabric
    )


# get args for env config
env_cfg.enable_wandb = args_cli.enable_wandb # type: ignore
env_cfg.enable_store = args_cli.enable_store # type: ignore

# get scene config from json while "enable_json" is True
if args_cli.enable_json:
    
    # get scene config from given file while "scene_file" is not None
    if args_cli.scene_file is not None:
        scene_json_path = args_cli.scene_file
    # otherwise,get scene config accordding to "scene_cfg_entry_point"
    else:
        scene_cfg_entry_point = gym.spec(args_cli.task).kwargs.get("scene_cfg_entry_point")
        # resolve path to the scene config location
        mod_name, file_name = scene_cfg_entry_point.split(":") # type: ignore
        mod_path = os.path.dirname(importlib.import_module(mod_name).__file__) # type: ignore
        scene_json_path = os.path.join(mod_path, file_name)
    scene_json = open(scene_json_path, 'r')
    scene_json = json.loads(scene_json.read())
    scene = scene_cfg(scene_json)
else:
    scene_cfg_entry_point = gym.spec(args_cli.task).kwargs.get("scene_cfg_entry_point")
    mod_name, attr_name = scene_cfg_entry_point.split(":") # type: ignore
    mod = importlib.import_module(mod_name)
    scene = getattr(mod, attr_name)

# 
scene.num_envs = args_cli.num_envs
# change scene attr of env config
env_cfg.scene = scene

env = gym.make(args_cli.task, cfg=env_cfg)

env.reset()

# env = type(TPEnv)env
# print(isinstance(env,TPEnv))
# env = TPEnvWrapper(env) # type: ignore
# wrap around environment for rl-games
# env = TPEnvDataCollectWrapper(env) # type: ignore
# env.run()
# pass
# 

while(True):
    #
    env.step(torch.zeros(1))
    # for i in range(env_cfg.decimation):
        
    # env.() 