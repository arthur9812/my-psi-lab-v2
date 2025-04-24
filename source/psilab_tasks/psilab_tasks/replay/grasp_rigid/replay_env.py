# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Common Modules  """ 
from __future__ import annotations
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
from psilab_tasks.teleoperation.grasp_rigid.scenes.room_scene_cfg import TASK_GRASP_RIGID_SCENE_CFG
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
    scene = TASK_GRASP_RIGID_SCENE_CFG # type: ignore



class ReplayEnv(RPEnv):

    cfg: ReplayEnvCfg

    def __init__(self, cfg: ReplayEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        # ############### initiallize variables ###################
        self._arm_joint_num = 7
        self._hand_real_joint_num = 6
        self._episodes = 0
        self._start_time= time.time()

        self.robot = self.scene.robots["robot1"]
        self.visualizer = self.scene.visualizer
        self.target = self.scene.rigid_objects["bottle"]

        pass
       

  
    def step(self,actions):
        #
        self.sim_step()
        # for i in range(self.cfg.sample_step):
            
        #
        # if self.vuer.bRecording:
        #     if self._data=={}:
        #         self.create_empty_data()

        #     self.parse_step_data()


        return super().step(actions)
        
    def sim_step(self):

        super().sim_step()

        # self.vuer.veur_step()

        # # 自动判断是否开始录制
        # if self.vuer.bControl and not self.vuer.bRecording:
        #     bPrepared = True
        #     # 开启录制条件1：右手位置与控制输入差距小于阈值
        #     state_c = self.vuer.right_wrist_pose + self.robot.data.body_link_state_w[0][0][:7]
        #     state = self.robot.data.body_link_state_w[0][self.robot.ik_controllers["arm2"].eef_link_index]
        #     delta_pos_norm = torch.norm((state[0:3]-state_c[0:3]),p=2)
        #     if delta_pos_norm>0.1:
        #         bPrepared = bPrepared and False
        #     # 开启录制条件2: 右手位姿速度小于阈值
        #     pos_vel = self.robot.data.body_state_w[0][self.robot.ik_controllers["arm2"].eef_link_index][7:10]
        #     angle_vel = self.robot.data.body_state_w[0][self.robot.ik_controllers["arm2"].eef_link_index][10:]
        #     pos_vel_norm = torch.norm(pos_vel,p=2)
        #     angle_vel_norm = torch.norm(angle_vel,p=2)
        #     # print(f"delta_pos: {delta_pos_norm}, pos_vel: {pos_vel_norm}, angle_vel: {angle_vel_norm}")
        #     if pos_vel_norm>0.02 or pos_vel_norm==0 or angle_vel_norm>0.02 or angle_vel_norm == 0:
        #         bPrepared = bPrepared and False
        #     #
        #     if bPrepared:
        #         self.vuer.bRecording = True
        #         #
        #         self.create_empty_data()
        #         # wrapped_env.start_record()
        #         self.voice.say("开始录制")

        # # Vuer Control mode
        # if self.vuer.bControl:
        #     # set ik command for arm
        #     self.scene.robots["robot1"].set_ik_command({
        #         "arm1": self.vuer.left_wrist_pose,
        #         "arm2": self.vuer.right_wrist_pose,
        #     })
        #     # set joint target for hand
        #     self.scene.robots["robot1"].set_joint_position_target(
        #         self.vuer.left_hand_joint_pos,
        #         self.scene.robots["robot1"].actuators["hand1"].joint_indices # type: ignore
        #     )
        #     self.scene.robots["robot1"].set_joint_position_target(
        #         self.vuer.right_hand_joint_pos,
        #         self.scene.robots["robot1"].actuators["hand2"].joint_indices # type: ignore
        #     )
        
        # # finished
        # if self.vuer.bFinished:
        #     self.save_h5()
        #     self.save_scene_cfg()
        #     self.reset()

        # # set image of vuer from sim
        # image_left = (self.scene.cameras["eye_left"].data.output["rgb"])[0]
        # image_right = (self.scene.cameras["eye_right"].data.output["rgb"])[0]
        # self.vuer.set_camera_image(image_left,image_right)

        # # 根据Vuer输出的头部姿态更新左右眼相机位姿态
        # # Tips: Vuer输出相对头部位姿，需要结合robot位置计算世界坐标系位置
        # robot_pos = self.scene.robots["robot1"].data.root_link_pos_w[0,:3]
        # self.scene.cameras["eye_left"].set_world_poses(self.vuer.left_eye_pose[0:3].add(robot_pos).unsqueeze(0),self.vuer.left_eye_pose[3:].unsqueeze(0),convention="world")
        # self.scene.cameras["eye_right"].set_world_poses(self.vuer.right_eye_pose[0:3].add(robot_pos).unsqueeze(0),self.vuer.right_eye_pose[3:].unsqueeze(0),convention="world")

   

    def _reset_idx(self, env_ids: torch.Tensor | None):

        # 
        super()._reset_idx(env_ids) # type: ignore


        # run 100 step until all rigid is static
        # for i in range(50):
        #     self.sim.step(render=True)
        #     self.scene.update(dt=self.physics_dt)

        

        

       

