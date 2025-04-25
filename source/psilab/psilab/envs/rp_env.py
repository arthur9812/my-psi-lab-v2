# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from __future__ import annotations
from typing import Any

""" Common Modules  """ 
import h5py
import torch
import numpy
import cv2
import matplotlib.pyplot as plt
""" Isaac Lab Modules  """ 
from isaaclab.envs.common import VecEnvStepReturn

""" Psi Lab Modules  """ 
from psilab.envs.rl_env import RLEnv
from psilab.envs.rp_env_cfg import RPEnvCfg


class RPEnv(RLEnv):
    """The replay environment class."""

    def __init__(self, cfg: RPEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)
        #
        self.cfg = cfg
        # get h5 file
        self._hdf5_file = h5py.File(cfg.hdf5_file, 'r')
        # 
        self._step = 0

        # fake state which is useless
        self._obs_zero = {
            "policy":torch.zeros((self.num_envs,self.cfg.observation_space),device=self.device), # type: ignore
            "critic":torch.zeros((self.num_envs,self.cfg.observation_space),device=self.device) # type: ignore
        }
        self._reward_zero = torch.zeros(self.num_envs,device=self.device) # type: ignore
        self._reset_zero = torch.tensor([0 for i in range(self.num_envs)], device=self.device)
        self._dones_zero = torch.tensor([0 for i in range(self.num_envs)], device=self.device)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:

        #
        self.sim_step()

        # return observations, rewards, resets and extras
        return self._obs_zero, self._reward_zero, self._reset_zero, self.reset_time_outs, dict()

    def sim_step(self):
        
        # repeat while replay finish
        if self._step >= self._hdf5_file["/timestamps"].len(): # type: ignore
            self._step = 0

        # robots
        for robot_name in list(self._hdf5_file["/robots"].keys()): # type: ignore
            # joints
            for joint_group_name in self._hdf5_file["/robots/"+robot_name+"/extra/joint_name"]:
                if joint_group_name=="all":
                    continue
                # 
                joint_pos = torch.tensor(self._hdf5_file["/robots/"+robot_name+"/"+joint_group_name + "_pos"][:][self._step],device="cuda:0").unsqueeze(0)# type: ignore
                joint_vel = torch.tensor(self._hdf5_file["/robots/"+robot_name+"/"+joint_group_name + "_vel"][:][self._step],device="cuda:0").unsqueeze(0)# type: ignore
                joint_indexs = self._hdf5_file["/robots/"+robot_name+"/extra/joint_index/"+joint_group_name][:].tolist() # type: ignore
                self.scene.robots[robot_name].write_joint_state_to_sim(joint_pos,joint_vel,joint_indexs)
            # TODO: add cameras image replay
            # BUG: not sure why camera names turn into bytes and other str in H5 is right
            # BUG: opencv imshow will freeze all!!!!
            # image = None
            # for camera_name in self._hdf5_file["/robots/"+robot_name+"/extra/cameras"]:
            #     if image is None:
            #         image = self._hdf5_file["/robots/"+robot_name+"/"+camera_name.decode("utf-8")][:][self._step] # type: ignore
            #     else:
            #         image = self._hdf5_file["/robots/"+robot_name+"/"+camera_name.decode("utf-8")][:][self._step] # type: ignore
            #         # image = numpy.concatenate((image, image2), axis=0)   # type: ignore # axis=0 按垂直方向，axis=1 按水平方向
            # if image is not None:
            #     image = self._hdf5_file["/robots/robot/arm2_camera.rgb"][:][self._step]
            #     # image = cv2.Mat(self._hdf5_file["/robots/robot/arm1_camera.rgb"][:][self._step])
            #     cv2.imshow("xx",image)
            #     cv2.waitKey(30)
            #     # plt.imshow("xx",image) # type: ignore
            #     # plt.pause(0.001)
            # pass
        # rigid object
        for object_name in list(self._hdf5_file["/rigid_objects"].keys()): # type: ignore
            state = torch.cat((
                torch.tensor(self._hdf5_file["/rigid_objects/"+object_name][:][self._step],device="cuda:0"),# type: ignore
                torch.zeros(6,device="cuda:0")),0).unsqueeze(0)

            self.scene.rigid_objects[object_name].write_root_state_to_sim(state)
        #
        self._step+=1

        # robot step to compute ik and ..., to set joint target
        for robot in self.scene.robots.values():
            robot.step()
        
        # set actions into simulator
        self.scene.write_data_to_sim()

        # simulate
        self.sim.step(render=True)
        # render between steps only if the GUI or an RTX sensor needs it
        # note: we assume the render interval to be the shortest accepted rendering interval.
        #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
        # if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
        #     self.sim.render()
        # update buffers at sim dt
        self.scene.update(dt=self.physics_dt)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        # reset scene
        self.scene.reset()
        return super().reset()



    """
    Functions for RL which is useless in Tele Operarion Env
    """
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        pass

    def _get_observations(self) -> dict:
        return self._obs_zero

    def _get_rewards(self) -> torch.Tensor:
       return self._reward_zero

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._reset_zero,self._dones_zero

