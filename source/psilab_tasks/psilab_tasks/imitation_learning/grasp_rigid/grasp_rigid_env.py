# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0



""" Python Modules  """ 
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar

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
from psilab_tasks.teleoperation.grasp_rigid.scenes.room_scene_cfg import TASK_GRASP_RIGID_SCENE_CFG
from psilab.eval.grasp_rigid import eval_success,eval_fail
from psilab.utils.data_collect_utils import create_empty_data,parse_step_data,save_data
from psilab.utils.wandb_utils import WandbLog
from psilab.utils.dp_utils import load_diffusion_policy_model,process_image
@configclass
class GraspRigidEnvCfg(ILEnvCfg):
    """Configuration for Rl environment."""

    # fake params
    episode_length_s = 1 * 210 / 60.0
    decimation = 2
    action_scale = 0.5
    action_space = 13
    observation_space = 130
    state_space = 130

    # 
    sample_step = 2

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
    scene = TASK_GRASP_RIGID_SCENE_CFG

    ouput_folder = OUTPUT_DIR + "/il/"


class GraspRigidEnv(ILEnv):

    cfg: GraspRigidEnvCfg

    def __init__(self, cfg: GraspRigidEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        # ############### initiallize variables ###################
        self._start_time= time.time()
        self._num = 0
        self._num_success = 0

        self.robot = self.scene.robots["robot1"]
        self.visualizer = self.scene.visualizer
        self.target = self.scene.rigid_objects["bottle"]

        # joint limit
        self._joint_limit_lower = self.robot.data.joint_limits[:,:,0].clone()
        self._joint_limit_upper = self.robot.data.joint_limits[:,:,1].clone()

        # load policy
        # checkpoint path
        policy_ckpt = "/home/admin01/Work/00-DiffusionPolicy/diffusion_policy/data/outputs/2025.04.20/13.08.28_train_diffusion_transformer_timm_dex_image_single_demo/checkpoints/epoch=6830-train_loss=0.002.ckpt"

        # load policy
        self.base_policy = load_diffusion_policy_model(
            policy_ckpt
        ).to("cuda:0")
   
  
    def step(self,actions):
        # policy model step

        for i in range(self.cfg.decimation):
            # record state
            if self.cfg.save_data:
                if self._data=={}:
                    create_empty_data(self,self.cfg)
                parse_step_data(self._data,self,self.cfg)
            #
            for j in range(self.cfg.sample_step):
                # simulator step
                self.sim_step()
            # get obs for policy

                
        return super().step(actions)
        
    def sim_step(self):

        

        # 构建当前观察字典
        eef_state = self.robot.data.body_link_state_w[0,self.robot.ik_controllers["arm2"].eef_link_index,:7].clone()
        eef_state[:3] -= self.robot.data.root_state_w[0,:3]
        
        # import matplotlib.pyplot as plt
        # image =self.robot.cameras["arm2_camera"].data.output["rgb"][0,:,:,:]
        # plt.imshow(image.cpu())
        # plt.pause(0.001)
        

        current_obs = {
            'base_camera_rgb': process_image(self.robot.cameras["base_camera"].data.output["rgb"][0,:,:,:]),
            'arm2_camera_rgb': process_image(self.robot.cameras["arm2_camera"].data.output["rgb"][0,:,:,:]),
            'arm2_pos': self.robot.data.joint_pos[0,self.robot.actuators["arm2"].joint_indices].unsqueeze(0),
            'hand2_pos': self.robot.data.joint_pos[0,self.robot.actuators["hand2"].joint_indices][:6].unsqueeze(0),
            'arm2_eef_pose': eef_state.unsqueeze(0),
        }


        # 添加批次维度
        current_obs = {k: v.unsqueeze(0) for k, v in current_obs.items()}

        # policy 推理
        with torch.no_grad():
            base_act_seq = self.base_policy.predict_action(current_obs)['action']
            action = base_act_seq.squeeze(0)[0]
        # set action
        action_lab = self.robot.data.default_joint_pos

        # 
        real_joint_index = self.robot.actuators["hand2"].joint_indices[:6] # type: ignore
        virtual_joint_index = self.robot.actuators["hand2"].joint_indices[6:] # type: ignore

        real_joint_pos_target_norm = norm(
            action[7:],
            self._joint_limit_lower[:,real_joint_index],
            self._joint_limit_upper[:,real_joint_index]
        )

        # 根据归一化结果和映射，修改联动关节
        action_virtual = real_joint_pos_target_norm[:,1:6] * (self._joint_limit_upper[:,virtual_joint_index] - self._joint_limit_lower[:,virtual_joint_index]) + self._joint_limit_lower[:,virtual_joint_index]


        # set target
        self.robot.set_joint_position_target(action[:7],self.robot.actuators["arm2"].joint_indices) # type: ignore

        # self.robot.set_joint_position_target(
        #     torch.cat((action[7:],action_virtual.squeeze(0)),dim=0),
        #     self.robot.actuators["hand2"].joint_indices) # type: ignore

        super().sim_step()
        
        # 判断任务成功或失败
        # # 失败判断
        # if eval_fail(
        #     self.scene.robots["robot1"],
        #     self.scene.rigid_objects["bottle"],
        #     contact_sensors, # type: ignore
        #     ): 
        #     print("Failed")
        #     self.reset()
        
        # # 成功判断
        # if eval_success(
        #     self.scene.robots["robot1"],
        #     self.scene.rigid_objects["bottle"],
        #     contact_sensors, # type: ignore
        #     0.3): 
        #     print("Success")
        #     save_data(self._data,self.cfg)
        #     self.reset()
        #     # 
        #     # record_count+=1

    def reset(self,seed: int | None = None, options: dict[str, Any] | None = None):
         # 打印测试结果
    # print(f"测试次数: {test_num} 次， 成功次数：{succes_num} 次, Policy成功率：{succes_num / test_num}")
        # self.vuer.reset()
        # run 100 step until all rigid is static
        # for i in range(50):
        #     self.sim.step(render=True)
        #     self.scene.update(dt=self.physics_dt)
        return super().reset()

       
# 将数据根据上下限制归一化至 [0,1]
@torch.jit.script
def norm(x, lower, upper):
    return (x-lower)/(upper-lower)
        

        

       

