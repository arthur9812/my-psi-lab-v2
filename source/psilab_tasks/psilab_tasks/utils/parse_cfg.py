# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-24
# Vesion: 1.0

""" Common Modules  """ 
import os
import json
import importlib
import gymnasium as gym

""" Isaac Lab Modules  """ 
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg

""" Psi Modules  """ 
from psilab.utils.config_utils import scene_cfg
from psilab.scene.sence_cfg import SceneCfg
from psilab.envs.rl_env_cfg import RLEnvCfg
from psilab.envs.tp_env_cfg import TPEnvCfg
from psilab.envs.il_env_cfg import ILEnvCfg
from psilab.envs.rp_env_cfg import RPEnvCfg

def parse_scene_cfg(
        task_name: str,
        enable_json: bool,
        json_file: str | None = None,
        num_envs: int | None = None)->SceneCfg:

    # get scene config from json while "enable_json" is True
    if enable_json:
        
        # get scene config accordding to "scene_cfg_entry_point" while "scene_file" is None
        if json_file is None:
            scene_cfg_entry_point = gym.spec(task_name).kwargs.get("scene_cfg_entry_point")
            # resolve path to the scene config location
            mod_name, file_name = scene_cfg_entry_point.split(":") # type: ignore
            mod_path = os.path.dirname(importlib.import_module(mod_name).__file__) # type: ignore
            json_file = os.path.join(mod_path, file_name)
        # get
        scene_json = open(json_file, 'r') # type: ignore
        scene_dict = json.loads(scene_json.read())
        scene = scene_cfg(scene_dict)
    else:
        scene_cfg_entry_point = gym.spec(task_name).kwargs.get("scene_cfg_entry_point")
        mod_name, attr_name = scene_cfg_entry_point.split(":") # type: ignore
        mod = importlib.import_module(mod_name)
        scene = getattr(mod, attr_name)
    #
    scene.num_envs = num_envs # type: ignore
    #
    return scene


def parse_rl_env_cfg(
        env_cfg,
        seed: int | None = None,
        enable_wandb: bool = False,
        enable_output: bool = False,
        ouput_folder: str | None = None,
        sample_step: int = 1)->RLEnvCfg:
    # 
    env_cfg.seed = seed # type: ignore
    env_cfg.enable_wandb = enable_wandb # type: ignore
    env_cfg.enable_output = enable_output # type: ignore
    env_cfg.sample_step = sample_step # type: ignore
    if ouput_folder is not None:
        env_cfg.ouput_folder = ouput_folder  # type: ignore

    #
    return env_cfg


def parse_rp_env_cfg(
        env_cfg,
        hdf5_file: str,
        json_file: str
        )->RPEnvCfg:
    # 
    env_cfg.hdf5_file = hdf5_file
    env_cfg.json_file = json_file

    #
    return env_cfg