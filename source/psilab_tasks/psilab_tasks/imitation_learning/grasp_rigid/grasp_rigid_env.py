# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0



""" Python Modules  """ 
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar
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
from psilab import OUTPUT_DIR
from psilab.envs.il_env import ILEnv 
from psilab.envs.il_env_cfg import ILEnvCfg
from psilab.utils.timer_utils import Timer
from psilab.eval.grasp_rigid import eval_success,eval_fail
from psilab.utils.data_collect_utils import create_data_buffer,parse_data,save_data
from psilab.utils.wandb_utils import WandbLog
# from psilab.utils.global_variants import GlobalVariant
from psilab.utils.dp_utils import load_diffusion_policy_model,process_image

@configclass
class GraspRigidEnvCfg(ILEnvCfg):
    """Configuration for Rl environment."""

    # fake params
    episode_length_s = 1 * 210 / 60.0
    decimation = 1
    action_scale = 0.5
    action_space = 13
    observation_space = 130
    state_space = 130

    # 
    sample_step = 1

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
            max_velocity_iteration_count = 4,
            bounce_threshold_velocity = 0.002,
            # enable_ccd=False,
            gpu_found_lost_pairs_capacity = 137401003
        ),
        render=RenderCfg(),

    )

    # scene config
    scene = MISSING # type: ignore

    # defualt ouput folder
    output_folder = OUTPUT_DIR + "/il"

class GraspRigidEnv(ILEnv):

    cfg: GraspRigidEnvCfg

    def __init__(self, cfg: GraspRigidEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        # episode
        self._episode = 0
        self._episode_success = 0

        # instances in scene
        self._robot = self.scene.robots["robot"]
        self._target = self.scene.rigid_objects["bottle"]

        # joint limit for compute later
        self._joint_limit_lower = self._robot.data.joint_limits[:,:,0].clone()
        self._joint_limit_upper = self._robot.data.joint_limits[:,:,1].clone()

        # load policy
        self.base_policy = load_diffusion_policy_model(
            self.cfg.policy
        ).to(self.device)

        # get timer
        self._timer = Timer()


        
    def step(self,actions):
        
        # import matplotlib.pyplot as plt
        # image =self._robot.cameras["base_camera"].data.output["rgb"][0,:,:,:]
        # plt.imshow(image.cpu())
        # plt.pause(0.001)

        # get obs for policy
        eef_link_index = self._robot.find_bodies("arm2_link7")[0][0]
        eef_state = self._robot.data.body_link_state_w[:,eef_link_index,:7].clone()
        eef_state[:,:3] -= self._robot.data.root_state_w[:,:3]
        #
        current_obs = {
            'base_camera_rgb': process_image(self._robot.cameras["base_camera"].data.output["rgb"][0,:,:,:]),
            'arm2_camera_rgb': process_image(self._robot.cameras["arm2_camera"].data.output["rgb"][0,:,:,:]),
            'arm2_pos': self._robot.data.joint_pos[:,self._robot.actuators["arm2"].joint_indices],
            'arm2_vel': self._robot.data.joint_vel[:,self._robot.actuators["arm2"].joint_indices],
            'hand2_pos': self._robot.data.joint_pos[:,self._robot.actuators["hand2"].joint_indices],
            'hand2_vel': self._robot.data.joint_vel[:,self._robot.actuators["hand2"].joint_indices],
            'arm2_eef_pos': eef_state[:,:3],
            'arm2_eef_quat': eef_state[:,3:7],

        }

        # 添加批次维度
        current_obs = {k: v.unsqueeze(0) for k, v in current_obs.items()}

        # policy model step
        with torch.no_grad():
            base_act_seq = self.base_policy.predict_action(current_obs)['action']
            self._action = base_act_seq.squeeze(0)[0]

        # 
        for i in range(self.cfg.decimation):
            # sim step
            self.sim_step()
            
        # output data
        if self.cfg.enable_output and self._sim_step_counter % self.cfg.sample_step == 0:
            parse_data(self._data,self,self.cfg)

        # time out
        if self._sim_step_counter % self.cfg.max_step == 0:
            self.reset()

        # # stop running
        # if self._episode >= self.cfg.max_episode:
        #     GlobalVariant().is_runing = False

        return super().step(actions)
        
    def sim_step(self):

        # 
        # real_joint_index = self._robot.actuators["hand2"].joint_indices[:6] # type: ignore
        # virtual_joint_index = self._robot.actuators["hand2"].joint_indices[6:] # type: ignore

        # real_joint_pos_target_norm = norm(
        #     self._action[7:],
        #     self._joint_limit_lower[:,real_joint_index],
        #     self._joint_limit_upper[:,real_joint_index]
        # )

        # # 根据归一化结果和映射，修改联动关节
        # action_virtual = real_joint_pos_target_norm[:,1:6] * (self._joint_limit_upper[:,virtual_joint_index] - self._joint_limit_lower[:,virtual_joint_index]) + self._joint_limit_lower[:,virtual_joint_index]


        # set target
        # aa = self._action.unsqueeze(0)
        # pass
        # action = torch.cat((,action_virtual),1)
        # aa = self._robot.actuators["arm2"].joint_indices
        # aaa = torch.tensor([ 0.5008, -0.9280,  1.7296, -2.4516,  0.3818,  0.6818, -0.4939], device='cuda:0')

        # aaa = torch.tensor([ 0.0, 0.0, 0.0,  0.0,  0.0, 0.0, 0.0], device='cuda:0')
        # self._robot.set_joint_position_target(self._action[:7],self._robot.actuators["arm2"].joint_indices) # type: ignore
        

        self._robot.set_joint_position_target(self._action[:7],self._robot.actuators["arm2"].joint_indices) # type: ignore
        self._robot.set_joint_position_target(self._action[7:],self._robot.actuators["hand2"].joint_indices) # type: ignore
            
        
        # self._robot.set_joint_position_target(
        #     torch.cat((self._action[7:].unsqueeze(0),action_virtual),dim=1),
        #     self._robot.actuators["hand2"].joint_indices) # type: ignore

        super().sim_step()
        
        # 
        contact_sensors = {
            "left_hand":self.scene.sensors["left_hand"],
            "right_hand":self.scene.sensors["right_hand"],
        }

        # 判断任务成功或失败
        # 失败判断
        if eval_fail(
            self.scene.robots["robot"],
            self.scene.rigid_objects["bottle"],
            contact_sensors, # type: ignore
            ): 
            print("Failed")
            self.reset()
        
        # 成功判断
        if eval_success(
            self.scene.robots["robot"],
            self.scene.rigid_objects["bottle"],
            contact_sensors, # type: ignore
            0.3): 
            print("Success")
            if self.cfg.enable_output:
                save_data(self._data,self.cfg)

            self._episode_success += 1

            # record_time = self._timer.run_time() /60.0
            # record_rate = self._episode_success / record_time
            #   
            print(f"Policy Success Rate: {self._episode_success/self._episode * 100} %")

            self.reset()

        self._sim_step_counter += 1
       

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        # 
        self._episode += 1
        #
        return super().reset()

# 将数据根据上下限制归一化至 [0,1]
@torch.jit.script
def norm(x, lower, upper):
    return (x-lower)/(upper-lower)
        

        

       

