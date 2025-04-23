# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from __future__ import annotations

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
from psilab.envs.tp_env import TPEnv 
from psilab.envs.tp_env_cfg import TPEnvCfg
from psilab.configs.scenes.task_grasp_rigid import TASK_GRASP_RIGID_SCENE_CFG
# from psila.assets.realman_inspire_no_camera import REALMAN_INSPIRE_NO_CAMERA_CFG
# from psi_rl import PSI_RL_USD_ASSET_DIR
from psilab.utils.wandb_utils import WandbLog

from psilab.configs.device.vuer_psi_dc_01 import VUER_PSI_DC_01_CFG
from psilab import OUTPUT_DIR
from psilab.utils.data_collect_utils import create_empty_data,parse_step_data,save_data

from psilab.eval.grasp_rigid import eval_success,eval_fail

@configclass
class GraspRigidEnvCfg(TPEnvCfg):
    """Configuration for Rl environment."""

    # fake params
    episode_length_s = 1 * 210 / 60.0
    decimation = 2
    action_scale = 0.5
    action_space = 13
    observation_space = 130
    state_space = 130

    # 
    sample_step = 1

    # device
    device_cfg = VUER_PSI_DC_01_CFG

    ouput_folder = OUTPUT_DIR + "/tp/"

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
            max_position_iteration_count = 255,
            max_velocity_iteration_count = 4,
            bounce_threshold_velocity = 0.002,
            enable_ccd=True,
            enable_enhanced_determinism=True,
            gpu_found_lost_pairs_capacity = 137401003
        ),
        render=RenderCfg(),
    )

    # scene config
    scene = TASK_GRASP_RIGID_SCENE_CFG


class GraspRigidEnv(TPEnv):

    cfg: GraspRigidEnvCfg

    def __init__(self, cfg: GraspRigidEnvCfg, render_mode: str | None = None, **kwargs):

        #
        super().__init__(cfg, render_mode, **kwargs)

        # ############### initiallize variables ###################
        self._arm_joint_num = 7
        self._hand_real_joint_num = 6
        self._episodes = 0
        self._start_time= time.time()

        self.robot = self.scene.robots["robot1"]
        self.visualizer = self.scene.visualizer
        self.target = self.scene.rigid_objects["bottle"]

        #
        self._record_count=0

        self.vuer.start()
        #
        # joint limit
        self._joint_limit_lower = self.robot.data.joint_limits[:,:,0].clone()
        self._joint_limit_upper = self.robot.data.joint_limits[:,:,1].clone()
        #
        # hand real joint index
        self._hand_real_joint_index_left = self.robot.actuators["hand1"].joint_indices[:6] # type: ignore
        self._hand_real_joint_index_right = self.robot.actuators["hand2"].joint_indices[:6] # type: ignore
        # hand virtual joint index
        self._hand_virtual_joint_index_left = self.robot.actuators["hand1"].joint_indices[6:] # type: ignore
        self._hand_virtual_joint_index_right = self.robot.actuators["hand2"].joint_indices[6:] # type: ignore
        #
        pass
       
   
    def step(self,actions):
        # 
        for i in range(self.cfg.sample_step):
            self.sim_step()
        #
        if self.vuer.bRecording:
            if self._data=={}:
                create_empty_data(self,self.cfg)

            parse_step_data(self._data,self,self.cfg)
            self._data["cameras"]["eye_left.rgb"]=[]
            self._data["cameras"]["eye_right.rgb"]=[]

        return super().step(actions)
        
    def sim_step(self):
        


        # print(self.scene.robots["robot1"].actuators["hand1"].joint_indices) # type: ignore)
        # self.vuer.veur_step()

        contact_sensors = {
            "left_hand":self.scene.sensors["left_hand"],
            "right_hand":self.scene.sensors["right_hand"],
        }

        # 开始录制,自动判断成功失败
        if self.vuer.bRecording:
            pass
            # 失败判断
            if eval_fail(
                self.scene.robots["robot1"],
                self.scene.rigid_objects["bottle"],
                contact_sensors, # type: ignore
                ): 
                print("Failed")
                self.reset()

            # 成功判断
            if eval_success(
                self.scene.robots["robot1"],
                self.scene.rigid_objects["bottle"],
                contact_sensors, # type: ignore
                0.3): 
                print("Success")
                save_data(self._data,self.cfg)
                self.reset()
                #
                self._record_count+=1
                record_stop_time = time.time()
                record_time =  (record_stop_time - self._start_time) /60.0
                record_rate = self._record_count / record_time
                #   

                print(f"采集时长: {record_time} 分钟")
                print(f"采集数据: {self._record_count} 条")
                print(f"采集效率: {record_rate} 条/分钟")

        # 自动判断是否开始录制
        if self.vuer.bControl and not self.vuer.bRecording:
            bPrepared = True
            # 开启录制条件1：右手位置与控制输入差距小于阈值
            state_c = self.vuer.right_wrist_pose + self.robot.data.body_link_state_w[0][0][:7]
            state = self.robot.data.body_link_state_w[0][self.robot.ik_controllers["arm2"].eef_link_index]
            delta_pos_norm = torch.norm((state[0:3]-state_c[0:3]),p=2)
            if delta_pos_norm>0.1:
                bPrepared = bPrepared and False
            # 开启录制条件2: 右手位姿速度小于阈值
            pos_vel = self.robot.data.body_state_w[0][self.robot.ik_controllers["arm2"].eef_link_index][7:10]
            angle_vel = self.robot.data.body_state_w[0][self.robot.ik_controllers["arm2"].eef_link_index][10:]
            pos_vel_norm = torch.norm(pos_vel,p=2)
            angle_vel_norm = torch.norm(angle_vel,p=2)
            # print(f"delta_pos: {delta_pos_norm}, pos_vel: {pos_vel_norm}, angle_vel: {angle_vel_norm}")
            if pos_vel_norm>0.02 or pos_vel_norm==0 or angle_vel_norm>0.02 or angle_vel_norm == 0:
                bPrepared = bPrepared and False
            #
            if bPrepared:
                self.vuer.bRecording = True
                #
                # self.create_empty_data()
                # wrapped_env.start_record()
                self.voice.say("开始录制")

        # Vuer Control mode
        if self.vuer.bControl:
            # set ik command for arm
            self.scene.robots["robot1"].set_ik_command({
                "arm1": self.vuer.left_wrist_pose,
                "arm2": self.vuer.right_wrist_pose,
            })
            #
            # virtual tendon
            hand_real_joint_pos_target_left_norm = norm(
                self.vuer.left_hand_joint_pos[:6],
                self._joint_limit_lower[:,self._hand_real_joint_index_left],
                self._joint_limit_upper[:,self._hand_real_joint_index_left]
            )
            hand_real_joint_pos_target_right_norm = norm(
                self.vuer.right_hand_joint_pos[:6],
                self._joint_limit_lower[:,self._hand_real_joint_index_right],
                self._joint_limit_upper[:,self._hand_real_joint_index_right]
            )

            # 根据归一化结果和映射，修改联动关节
            self.vuer.left_hand_joint_pos[6:] = hand_real_joint_pos_target_left_norm[:,1:6] * (self._joint_limit_upper[:,self._hand_virtual_joint_index_left] - self._joint_limit_lower[:,self._hand_virtual_joint_index_left]) + self._joint_limit_lower[:,self._hand_virtual_joint_index_left]
            self.vuer.right_hand_joint_pos[6:] = hand_real_joint_pos_target_right_norm[:,1:6] * (self._joint_limit_upper[:,self._hand_virtual_joint_index_right] - self._joint_limit_lower[:,self._hand_virtual_joint_index_right]) + self._joint_limit_lower[:,self._hand_virtual_joint_index_right]

            # set joint target for hand
            self.scene.robots["robot1"].set_joint_position_target(
                self.vuer.left_hand_joint_pos,
                self.scene.robots["robot1"].actuators["hand1"].joint_indices # type: ignore
            )
            self.scene.robots["robot1"].set_joint_position_target(
                self.vuer.right_hand_joint_pos,
                self.scene.robots["robot1"].actuators["hand2"].joint_indices # type: ignore
            )

        if self.vuer.bReset:
            self.vuer.reset()
            self.reset()
        # finishe
        # if self.vuer.bFinished:
        #     save_data(self._data,self.cfg)
        #     self.reset()

        # set image of vuer from sim
        image_left = (self.scene.cameras["eye_left"].data.output["rgb"])[0]
        image_right = (self.scene.cameras["eye_right"].data.output["rgb"])[0]
        self.vuer.set_camera_image(image_left,image_right)

        # 根据Vuer输出的头部姿态更新左右眼相机位姿态
        # Tips: Vuer输出相对头部位姿，需要结合robot位置计算世界坐标系位置
        robot_pos = self.scene.robots["robot1"].data.root_link_pos_w[0,:3]
        self.scene.cameras["eye_left"].set_world_poses(self.vuer.left_eye_pose[0:3].add(robot_pos).unsqueeze(0),self.vuer.left_eye_pose[3:].unsqueeze(0),convention="world")
        self.scene.cameras["eye_right"].set_world_poses(self.vuer.right_eye_pose[0:3].add(robot_pos).unsqueeze(0),self.vuer.right_eye_pose[3:].unsqueeze(0),convention="world")

        super().sim_step()

        
# 将数据根据上下限制归一化至 [0,1]
@torch.jit.script
def norm(x, lower, upper):
    return (x-lower)/(upper-lower)

