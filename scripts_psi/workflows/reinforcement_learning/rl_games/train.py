# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Arguments parse """
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates lego grasp task demo from gym.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--width", type=int, default=1920, help="Width of the viewport and generated images.")
parser.add_argument("--height", type=int, default=1080, help="Height of the viewport and generated images.")


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

# store args befor create app as it will pop some arg from args_cli
enable_cameras = args_cli.enable_cameras

# launch omniverse app
app_launcher = AppLauncher(args_cli)

""" Common Modules  """ 
import os
import sys
import math
import importlib
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

""" Psi RL Modules  """ 
# import psilab.tasks # noqa: F401
import psilab_tasks
from psilab.utils.config_utils import scene_cfg
# print('psilab_tasks.rl' in sys.modules)

# create env
env_cfg= parse_env_cfg(
    args_cli.task, 
    device=args_cli.device,
    num_envs=args_cli.num_envs,
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

# clear camera configs in scene while "enable_cameras" flag is True
if enable_cameras is False:
    scene.cameras_cfg ={}
    for robot_cfg in scene.robots_cfg.values():
        robot_cfg.cameras = {} # type: ignore

# create env
env = gym.make(args_cli.task, cfg=env_cfg)


# parse agent configuration
agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

# specify directory for logging experiments
log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"]) # type: ignore
log_root_path = os.path.abspath(log_root_path)
print(f"[INFO] Logging experiment in directory: {log_root_path}")
# specify directory for logging runs
log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))# type: ignore
# set directory into agent config
# logging directory path: <train_dir>/<full_experiment_name>
agent_cfg["params"]["config"]["train_dir"] = log_root_path # type: ignore
agent_cfg["params"]["config"]["full_experiment_name"] = log_dir # type: ignore

# dump the configuration into log-directory
dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

# read configurations about the agent-training
rl_device = agent_cfg["params"]["config"]["device"] # type: ignore
clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf) # type: ignore
clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf) # type: ignore

# wrap around environment for rl-games
env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions) # type: ignore


# register the environment to rl-games registry
# note: in agents configuration: environment name must be "rlgpu"
vecenv.register(
    "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
)
env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})


# set number of actors into agent config
agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs # type: ignore

# create runner from rl-games
runner = Runner(IsaacAlgoObserver())
runner.load(agent_cfg)

# reset the agent and env
runner.reset()

# train the agent
if args_cli.checkpoint is not None:
    runner.run({"train": True, "play": False, "sigma": None})
else:
    runner.run({"train": True, "play": False, "sigma": None})

# close the simulator
env.close()
