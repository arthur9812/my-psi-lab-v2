# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Common Modules  """ 
import torch

""" IsaacLab Modules  """ 
from  isaaclab.controllers.differential_ik import DifferentialIKController

""" PsiLab Modules  """ 
from psilab.controllers.differential_ik_cfg import DiffIKControllerCfg


class DiffIKController(DifferentialIKController):
    """psilab differential inverse kinematics controller."""

    eef_link_index:int = None # type: ignore
    """ The end effector link index of ik controller among all robot link """

    eef_jacobian_index:int = None # type: ignore
    """ The end effector link jacobian index """

    joint_index: list[int] = None # type: ignore
    """ The index of joints which be controlled by ik controller """

    def __init__(self, cfg: DiffIKControllerCfg, num_envs: int, device: str):
        super().__init__(cfg,num_envs,device)
        # overwrite cfg
        self.cfg = cfg

    def initialize_impl(self,robot): # type: ignore

        # 
        self.joint_index = robot.find_joints(self.cfg.joint_name)[0]
        # 
        self.eef_link_index = robot.find_bodies(self.cfg.eef_link_name)[0][0]
        # 
        self.eef_jacobian_index = self.eef_link_index - 1 
        # 
        
        
    def reset(self,robot):
        #
        super().reset()
        # 
        self.eef_pose_init = robot.data.body_link_state_w[0,self.eef_link_index,:7]
        # print(robot.data.body_link_state_w[0,16,:7])
        # print(robot.data.body_link_state_w[0,17,:7])

        # 
        self.eef_pose_init[:3] -= robot.data.root_link_pos_w[0,:3]
        #
        self.set_command(self.eef_pose_init)