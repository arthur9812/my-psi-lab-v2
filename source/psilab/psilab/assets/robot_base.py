# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Sequence

""" Common Modules  """ 
import torch


""" IsaacLab Modules  """ 
from isaaclab.assets.articulation import Articulation
from isaaclab.sensors.camera import CameraCfg,Camera,TiledCamera
from isaaclab.utils.math import ( 
    matrix_from_quat,
    quat_inv,
    subtract_frame_transforms,
)

""" PsiLab Modules  """ 
from psilab.controllers.differential_ik import DiffIKController

if TYPE_CHECKING:
    from .robot_base_cfg import RobotBaseCfg

class RobotBase(Articulation):
    """An robot asset class.
    An robot is an articulation with some ik controllers which can be easily controled by device and policy input 
    """
    cfg: RobotBaseCfg

    cameras : dict[str,Camera] = None # type: ignore
    tiled_cameras : dict[str,TiledCamera] = None # type: ignore

    ik_controllers : dict[str,DiffIKController] = None # type: ignore

    eef_links : dict[str,int] = None # type: ignore

    def __init__(self, cfg: RobotBaseCfg):
        """Initialize the Robot.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

        self.cameras = {}
        self.tiled_cameras = {}
        self.ik_controllers = {}
        self.eef_links = {}

    def _initialize_impl(self):

        super()._initialize_impl()
        #
        for ik_name,ik_cfg in self.cfg.diff_ik_controllers.items():
            #
            self.ik_controllers[ik_name] = DiffIKController(ik_cfg, num_envs=1, device=self.device)
            #
            self.ik_controllers[ik_name].initialize_impl(self)
            #
            # self.ik_controllers[ik_name].reset(self)
        #
        for eef_name,eef_link_name in self.cfg.eef_links.items():
            eef_index = self.find_bodies(eef_link_name)[0][0]
            self.eef_links[eef_name] = eef_index

    def reset(self, env_ids: Sequence[int] | None = None):
        """
            Reset robot, include joint, actuator, ik controllers
        """
        # reset articulation, which will reset actuator
        super().reset()
        # reset all joint state and target to default state
        self.set_joint_position_target(self.data.default_joint_pos.clone())
        self.write_joint_state_to_sim(
            position=self.data.default_joint_pos.clone(),
            velocity=torch.zeros(self.num_joints,device=self.device)
        )
        self.write_data_to_sim()   
        # print(self.data.joint_pos_target[0,:])
        #
        for ik_name,ik_cfg in self.cfg.diff_ik_controllers.items():
            self.ik_controllers[ik_name].reset(self)

        # self.data.root_link_pos_w
 
    def step(self):
        """
        Step for each simulation step, include all computation and set varibale values to data
        """
        pass
        # get root stae 
        root_pose_w = self.data.root_link_state_w[:, 0:7]
        base_rot = root_pose_w[:, 3:7]
        base_rot_matrix = matrix_from_quat(quat_inv(base_rot))
        # traverse ik controllers
        for controller in self.ik_controllers.values():
            jacobian = self.root_physx_view.get_jacobians()[:,controller.eef_jacobian_index, :, controller.joint_index]
            #
            ee_pose_w = self.data.body_link_state_w[:, controller.eef_link_index, :7]
            #
            jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
            jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
            #
            joint_pos = self.data.joint_pos[:, controller.joint_index]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            # compute the joint commands
            pos_target = controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            # 
            self.set_joint_position_target(pos_target,controller.joint_index)

    def set_ik_command(self,command: dict[str,torch.Tensor]):
        """
            Set command for all ik controllers
        """
        ik_command_keys = list(command.keys())
        for ik_name in self.cfg.diff_ik_controllers.keys():
            if ik_name in ik_command_keys:
                self.ik_controllers[ik_name].set_command(command[ik_name])

