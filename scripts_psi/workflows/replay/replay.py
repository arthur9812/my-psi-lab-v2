# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0



""" Arguments parse """
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates lego grasp task demo from gym.")

parser.add_argument("--task", type=str, default="", help="Name of the task.")
# parser.add_argument("--device", type=str, default="cuda:0", help="Name of the task.")
parser.add_argument("--h5_file", type=str, default="", help="Name of the task.")
parser.add_argument("--config_json", type=str, default="", help="Name of the task.")
parser.add_argument("--output_dir", type=str, default=None, help="Path to model checkpoint.")


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
from psilab.utils.config_utils import scene_cfg
# from psilab.envs.tp_env import TPEnvDataCollectWrapper
# create env
env_cfg= parse_env_cfg(
    args_cli.task, 
    device=args_cli.device,
    num_envs=1,
    # use_fabric=not args_cli.disable_fabric
    )

env_cfg.h5_file = args_cli.h5_file  # type: ignore

scene_json_path = "/".join(env_cfg.h5_file.split("/")[:-1]) + "/" +"_".join(env_cfg.h5_file.split("/")[-1].split("_")[:2])+"_scene_config.json" # type: ignore

scene_json = open(scene_json_path, 'r')
scene_json = json.loads(scene_json.read())
env_cfg.scene = scene_cfg(scene_json)

import time
# aaa  = scene_cfg.from_dict(scene_json)
env = gym.make(args_cli.task, cfg=env_cfg)
time.sleep(1)

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