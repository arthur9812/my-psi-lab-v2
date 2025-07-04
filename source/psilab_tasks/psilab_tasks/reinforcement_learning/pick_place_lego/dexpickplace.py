# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: 
# Date: 2025-05-27
# Vesion: 1.0

""" Common Modules  """ 
from __future__ import annotations
import torch
from torch.linalg import svd
import warnings
from datetime import datetime
from typing import Tuple

""" Isaac Sim Modules  """ 
import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate

""" Isaac Lab Modules  """ 
from isaaclab.sim import SimulationCfg, PhysxCfg, RenderCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate


""" Psi Lab Modules  """
from psilab import OUTPUT_DIR
from psilab.envs.rl_env import RLEnv 
from psilab.envs.rl_env_cfg import RLEnvCfg
from psilab.utils.wandb_utils import WandbLog
from psilab.utils.timer_utils import Timer
from psilab.utils.data_collect_utils import save_data,save_data_muilt_env
from psilab.eval.grasp_rigid import eval_fail, eval_success
import psilab.utils.color_print as cp

import numpy as np
@configclass
class DexPickPlaceEnvCfg(RLEnvCfg):
    """Configuration for RL environment."""

    # params
    max_episode_length = 384
    # max_episode_length = 1024
    episode_length_s = 1.0 * max_episode_length / 60.0
    decimation = 2
    action_scale = 0.5
    action_space = 13
    observation_space = 181
    state_space = 181

    # other params from gym
    arm_hand_dof_speed_scale = 20.0
    vel_obs_scale = 0.2
    act_moving_average = 0.8
    env_id_print_data = 0 # index of env to print status
    lift_height_target = 0.3 # target lift height target

    lift_target_delta = [0.0, 0.2, 0.3]
    lift_targets = [[0.5,-0.105,0.8], # grasp
                    [0.5,0.0,1.0], # up
                    # [0.5,0.105,1.0], # right
                    [0.5,0.105,0.75], # down
                    [0.5,0.105,0.75], # standby
                    [0.0,0.0,0.0]] # ending state
    # lift_targets = [[0.5,-0.105,1.1]]


    reward_func = [{'standby': False, 'grasp': True, 'position': False, 'orientation': False},
                    {'standby': False, 'grasp': True, 'position': True, 'orientation': False},
                    # {'standby': False, 'grasp': True, 'position': True, 'orientation': True},
                    {'standby': False, 'grasp': True, 'position': True, 'orientation': False},
                    {'standby': True, 'grasp': False, 'position': True, 'orientation': False},
                    {'standby': True, 'grasp': False, 'position': False, 'orientation': False}] # ending state

    # simulation config
    sim: SimulationCfg = SimulationCfg(
        dt = 1 / 120, 
        render_interval=decimation,
        physx = PhysxCfg(
            solver_type = 1, # 0: pgs, 1: tgs
            max_position_iteration_count = 32,
            max_velocity_iteration_count = 0,
            bounce_threshold_velocity = 0.002,
            enable_ccd=True,
            gpu_max_rigid_patch_count = 4096 * 4096,
            gpu_collision_stack_size = 1600000000,
            gpu_found_lost_pairs_capacity = 137401003,
            # gpu_total_aggregate_pairs_capacity=5196400

        ),
        render=RenderCfg(),

    )

    # defualt ouput folder
    output_folder = OUTPUT_DIR + "/rl"

class DexPickPlaceEnv(RLEnv):
    """GraspLego RL environment."""

    cfg: DexPickPlaceEnvCfg

    def __init__(self, cfg: DexPickPlaceEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        # ############### initiallize variables ###################
        self._arm_joint_num = 7
        self._hand_real_joint_num = 6
        self._episodes = 0

        # get instances in scene
        self._robot = self.scene.robots["robot"]
        self._target = self.scene.rigid_objects["target"]
        self._visualizer = self.scene.visualizer

        self._contact_sensors = {}
        for key in ["hand2_link_base",
                    "hand2_link_1_1",
                    "hand2_link_1_2",
                    "hand2_link_1_3",
                    "hand2_link_2_1",
                    "hand2_link_2_2",
                    "hand2_link_3_1",
                    "hand2_link_3_2",
                    "hand2_link_4_1",
                    "hand2_link_4_2",
                    "hand2_link_5_1",
                    "hand2_link_5_2"]:
            self._contact_sensors[key] = self.scene.sensors[key]

        # arm joint index
        self._arm_joint_index = [self._robot.find_joints(joint_name)[0][0] for joint_name in self._robot.actuators["arm2"].joint_names]
        # hand joint index
        self._hand_joint_index = [self._robot.find_joints(joint_name)[0][0] for joint_name in self._robot.actuators["hand2"].joint_names]
        self._hand_base_link_index = self._robot.find_bodies(["hand2_link_base"])[0][0]
        # hand real joint index
        self._hand_real_joint_index = self._hand_joint_index[:6]
        # hand virtual joint index
        self._hand_virtual_joint_index = self._hand_joint_index[6:]
        # finger tip link index
        self._finger_tip_index = [
                self._robot.find_bodies(link_name)[0][0] 
                for link_name in [
                    "hand2_link_1_4",
                    "hand2_link_2_3",
                    "hand2_link_3_3",
                    "hand2_link_4_3",
                    "hand2_link_5_3",
                ]
            ]
        # robot index
        # [note]: this index means real control index
        self._robot_index = self._arm_joint_index + self._hand_real_joint_index
        
        # joint limit
        self._joint_limit_lower = self._robot.data.joint_limits[:,:,0].clone()
        self._joint_limit_upper = self._robot.data.joint_limits[:,:,1].clone()
        # robot, object inital state
        self._curr_targets = self._robot.data.default_joint_pos.clone()
        self._prev_targets = self._robot.data.default_joint_pos.clone()
        self._target_init_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self._target_lift_pose = torch.zeros((self.num_envs, 3), device=self.device)
        self._final_target_pose = torch.tensor(self.cfg.lift_targets[-2], device=self.device).repeat(self.num_envs, 1)
        self._target_lift_height = self.cfg.lift_height_target * torch.ones(self.num_envs, device=self.device)
        self._target_lift_delta = torch.tensor(self.cfg.lift_target_delta, device=self.device).repeat(self.num_envs, 1)
        self._hand_init_pose = torch.zeros((self.num_envs, 3), device=self.device)
        self._hand_target_pose = torch.zeros((self.num_envs, 3), device=self.device)

        # unit tensors
        self._x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self._y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self._z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        
        # variables for rl training
        self._pre_distance_reward = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        self._pre_pose_reward = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        self._pre_lift_reward = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        self._pre_angle_reward = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        self._pre_energy = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        # metrics for task
        self._contacted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._successed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # subtasks
        self._subtask_index = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._task_playing = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        
        self.reward_func_active = {}
        for key, value in self.cfg.reward_func[0].items():
            self.reward_func_active[key] = torch.tensor(value, device=self.device).repeat(self.num_envs)

        # Print information of this environment
        print("Num envs: ", self.num_envs)
        print("Num bodies: ", self._robot.num_bodies)
        print("Num arm dofs: ", self._arm_joint_num)
        print("Num hand dofs: ", 6)
        print("hand_base_rigid_body_index: ", self._hand_base_link_index)
        
        self.extras = {'standby_reward': 0.0, 'dist_reward': 0.0, 'pose_reward': 0.0, 'lift_reward': 0.0, 'angl_reward': 0.0, 'orient_reward': 0.0, 'act_penalty': 0.0, 'success': 0.0}
        
        # initialize wandb
        if self.cfg.enable_wandb: 
            self._wandb = WandbLog()
            project = "PsiLab_v2.0_RL"
            name = "Pick-Place-Lego" + datetime.strftime(datetime.now(), '%m%d_%H%M%S')
            tags = ["successed cond", "subtask shift"]
            tags.append("standby reward")
            # tags.append("orientation reward")
            # tags.append("support polygon")
            # tags.append("force closure")
            tags.append("finger pointing")
            self._wandb.init_wandb(project, name, tags)

        # initialize Timer
        self._timer = Timer()

        # initiallize output count
        self._output_count = 0
   
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
    
    def _apply_action(self) -> None:
        '''        self.episodes += 1
        action range       : (-1,1)
        action index 0-6   : arm joint
        action index 7-12  : real hand joint (order: 1-1,2-1,3-1,4-1,5-1,1-2)
        action index 13-17 : fake hand joint (order: 1-3,2-2,3-2,4-2,5-2)
        '''
        # ============ control arm =============
        arm_targets = self._robot.data.joint_pos[:, self._arm_joint_index] + \
                      self.cfg.arm_hand_dof_speed_scale  * self.physics_dt * self.actions[:, :self._arm_joint_num]

        self._curr_targets[:, self._arm_joint_index] = saturate(
            arm_targets,
            self._joint_limit_lower[:, self._arm_joint_index],
            self._joint_limit_upper[:, self._arm_joint_index]
        )
        
        # ============ control hand ============
        self._curr_targets[:, self._hand_real_joint_index] = scale(
            self.actions[:, self._arm_joint_num:],
            self._joint_limit_lower[:, self._hand_real_joint_index],
            self._joint_limit_upper[:, self._hand_real_joint_index]
        )
        self._curr_targets[:, self._hand_real_joint_index] = (
            self.cfg.act_moving_average * self._curr_targets[:, self._hand_real_joint_index]
            + (1.0 - self.cfg.act_moving_average) * self._prev_targets[:, self._hand_real_joint_index]
        )
        self._curr_targets[:, self._hand_real_joint_index] = saturate(
            self._curr_targets[:, self._hand_real_joint_index],
            self._joint_limit_lower[:, self._hand_real_joint_index],
            self._joint_limit_upper[:, self._hand_real_joint_index]
        )
        
        self._prev_targets = self._curr_targets.clone()
        self._robot.set_joint_position_target(
            self._curr_targets[:, self._robot_index], joint_ids=self._robot_index
        ) 
        
    def step(self, action):
        # call super step first to apply action and sim step
        obs_buf, reward_buf, reset_terminated, reset_time_outs, extras = super().step(action)
        
        self._maker_visualizer()

        return obs_buf, reward_buf, reset_terminated, reset_time_outs, extras

    def _get_observations(self) -> dict:
        # implement fingertip force sensors
        # self.fingertip_force_sensors = self.robot.root_physx_view.get_link_incoming_joint_force()[:, self._finger_tip_index]
        
        self._get_full_observations()
        observations = {"policy": self._obs, "critic":self._obs}

        return observations
    
    def _get_rewards(self) -> torch.Tensor:
        standby_active = self.reward_func_active['standby']
        position_active = self.reward_func_active['position']
        grasp_active = self.reward_func_active['grasp']
        orientation_active = self.reward_func_active['orientation']
        standby_reward, distance_reward, pose_reward, angle_reward, lift_reward, orientation_reward, action_penalty = _compute_rewards(
            self.finger_thumb_state,
            self.finger_index_state,
            self.finger_middle_state,
            self.middle_point_state,
            self._hand_target_pose,
            self.hand_base_state[:, :3],
            self._target_init_pose[:, :7],
            self.object_state[:, :7],
            self.actions[:, :self._arm_joint_num],
            self._curr_targets[:, self._hand_real_joint_index],
            self._prev_targets[:, self._hand_real_joint_index],
            self._z_unit_tensor,
            self._target_lift_pose,
            standby_active,
            position_active,
            grasp_active,
            orientation_active,
        )
        # print(cp.LYH_DEBUG("pose_reward:"), cp.LYH_DEBUG(pose_reward.mean().item()),cp.LYH_DEBUG("dist reward:"), cp.LYH_DEBUG(distance_reward.mean().item()))
        # print(cp.red("self.arm_hand_target_pose:"), cp.red(self._arm_hand_target_pose[0].to('cpu').numpy().tolist()))
        total_reward = (standby_reward + distance_reward + pose_reward + lift_reward + angle_reward + orientation_reward - self._pre_energy) - action_penalty
        # print(cp.yellow("total reward:"), cp.yellow(total_reward.mean().item()))
        self.extras['standby_reward'] += (standby_reward.mean() - self._pre_standby_reward.mean())  # type: ignore
        self.extras['dist_reward'] += (distance_reward.mean() - self._pre_distance_reward.mean())  # type: ignore
        self.extras['pose_reward'] += (pose_reward.mean() - self._pre_pose_reward.mean())# type: ignore
        self.extras['lift_reward'] += (lift_reward.mean() - self._pre_lift_reward.mean())# type: ignore
        self.extras['angl_reward'] += (angle_reward.mean() - self._pre_angle_reward.mean())# type: ignore
        self.extras['orient_reward'] += (orientation_reward.mean() - self._pre_orientation_reward.mean())# type: ignore
        self.extras['act_penalty'] += action_penalty.mean() # type: ignore

        # update pre reward
        self._pre_standby_reward = standby_reward
        self._pre_distance_reward = distance_reward
        self._pre_pose_reward = pose_reward
        self._pre_lift_reward = lift_reward
        self._pre_angle_reward = angle_reward
        self._pre_orientation_reward = orientation_reward
        self._pre_energy = standby_reward + distance_reward + pose_reward + lift_reward + angle_reward + orientation_reward

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.async_reset:
            bfailed, self._contacted = eval_fail(self._target,self._contact_sensors,self._contacted)
            self._successed = eval_success(self._target, self._contact_sensors, self.cfg.lift_height_target)
            resets = self._successed & bfailed
        else:
            resets = torch.zeros(self.num_envs, device=self.device)
        
        # self._successed = (self._target.data.root_pos_w[:, 2] - self._target_init_pose[:, 2]) >= self.cfg.lift_height_target * 0.8
        self._successed = self._is_success()
        self.extras['success'] = self._successed.float().mean().item() * 100.0
        
        return resets, time_out

    def _is_success(self) -> torch.Tensor:
        return ~self._task_playing

    def _is_subtask_success(self, target_pos: torch.Tensor, object_pos: torch.Tensor, ) -> torch.Tensor:
        
        grasp_active = self.reward_func_active['grasp']
        grasp_check = (self._pre_distance_reward[:] > 0.7) & (self._pre_pose_reward[:] >= 6.5)
        grasp_success = torch.where(grasp_active, grasp_check, torch.ones_like(grasp_check, dtype=torch.bool))

        standby_active = self.reward_func_active['standby']
        standby_check = torch.norm(self.hand_base_state[:, :3] - self._hand_target_pose, p=2, dim=-1) < 0.05
        standby_success = torch.where(standby_active, standby_check, torch.ones_like(standby_check, dtype=torch.bool))

        position_active = self.reward_func_active['position']
        position_check = (torch.norm(object_pos - target_pos, p=2, dim=-1)) < 0.05
        position_success = torch.where(position_active, position_check, torch.ones_like(position_check, dtype=torch.bool))

        orientation_active = self.reward_func_active['orientation']
        orientation_check = _quat_sin2_loss(self._target_init_pose[:, 3:7], self.object_state[:, 3:7]) < 0.05
        orientation_success = torch.where(orientation_active, orientation_check, torch.ones_like(orientation_check, dtype=torch.bool))
        
        return grasp_success & standby_success & position_success & orientation_success

    def _is_task_failed(self) -> torch.Tensor:
        task_failed_env_ids = (self.object_state[:, 2] - self._target_init_pose[:, 2] > 0.05) & (self._pre_pose_reward <= 1.0) & (self._subtask_index > 0) & self.reward_func_active['grasp']
        task_failed_env_ids = torch.nonzero(task_failed_env_ids, as_tuple=False).squeeze(-1)
        return task_failed_env_ids

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if self.cfg.enable_output and self._data is not None:
            # get index of envs will save data
            # asynchronous reset
            if self.cfg.async_reset:
                env_save_list=[]
                for env_index in env_ids:
                    if not self.reset_time_outs[env_index]:
                        env_save_list.append(env_index)
                
            # synchronous reset
            else:
                env_save_list = []
                for i in range(self.scene.num_envs):
                    delta_z = self._target.data.root_com_pos_w[i,2] - self._target_init_pose[i,2]
                    if self._is_success()[i]:
                    # if abs(delta_z - self.cfg.lift_height_target)<0.1:
                        env_save_list.append(i)

            # save data 
            # print(env_save_list)
            if len(env_save_list)>0:
                print(env_save_list)
                # single env
                if self.scene.num_envs == 1:
                    # save_data(self._data,self.cfg)
                    pass
                # multi env
                elif self.scene.num_envs >1:
                    pass
                    save_data_muilt_env(self._data,self.cfg,env_save_list)
                else:
                    raise Exception(f"Save Data Error as {self.scene.num_envs} is incorrect") 

                self._output_count += len(env_save_list)
                # record_time = self._timer.run_time() /60.0
                # record_rate = self._output_count / record_time
                #   
                # print(f"采集时长: {record_time} 分钟")
                # print(f"采集数据: {self._output_count} 条")
                # print(f"采集效率: {record_rate} 条/分钟")
            #
            record_time = self._timer.run_time() /60.0
            print(f"时长: {record_time} 分钟")
            record_rate = self._episodes / record_time
            print(f"效率: {record_rate} 条/分钟")
            print(f"成功条数/总条数: {self._output_count}/{self._episodes} ")

        # update episodes
        # TODO: change this part code for self._episodes += len(env_ids)
        if self.cfg.async_reset:
            self._episodes += 1
        else:
            self._episodes += self.num_envs
        
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES # type: ignore
        super()._reset_idx(env_ids) # type: ignore
        
        ############ reset robot ################
        dof_pos = self._robot.data.default_joint_pos.clone()
        dof_vel = self._robot.data.default_joint_vel.clone()
        self._curr_targets[env_ids, :] = dof_pos[env_ids, :]
        self._prev_targets[env_ids, :] = dof_vel[env_ids, :]
        # TODO: check api params
        self._robot.set_joint_position_target(dof_pos, env_ids=env_ids) # type: ignore
        self._robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids) # type: ignore
        
        ############ reset object ################
        # reset_position_noise = 0.00
        # lego_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device) * reset_position_noise
        # lego_rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        # object_defaut_state = self._target.data.root_link_state_w[env_ids].clone()
        # object_defaut_state[:, 0:3].add_(lego_pos_noise)
        # # object_defaut_state[:, 0:3].add_(self.scene.env_origins[env_ids])
        # object_defaut_state[:, 3:7] = randomize_rotation(
        #     lego_rot_noise[:, 0], lego_rot_noise[:, 1], self._x_unit_tensor[env_ids], self._y_unit_tensor[env_ids]
        # )
        # object_defaut_state[:, 7:] = torch.zeros_like(self._target.data.root_link_state_w[env_ids, 7:])
        # self._target.write_root_link_pose_to_sim(object_defaut_state[:, :7], env_ids)    # type: ignore
        # self._target.write_root_com_velocity_to_sim(object_defaut_state[:, 7:], env_ids) # type: ignore
        
        for _ in range(30):
            self.sim.step(render=True)
            self.scene.update(dt=self.physics_dt)

        ############ last logs ################
        # wandb log
        if self.cfg.enable_wandb:
            self._wandb.set_data("subtask_index", self._subtask_index.float().mean().item())
            self._wandb.set_data("success", self._successed.float().mean().item() * 100.0)
            self._wandb.set_data("standby_reward", self.extras['standby_reward'])
            self._wandb.set_data("dist_reward", self.extras['dist_reward'])
            self._wandb.set_data("pose_reward", self.extras['pose_reward'])
            self._wandb.set_data("lift_reward", self.extras['lift_reward'])
            self._wandb.set_data("angl_reward", self.extras['angl_reward'])
            self._wandb.set_data("orient_reward", self.extras['orient_reward'])
            self._wandb.set_data("act_penalty", self.extras['act_penalty'])
            self._wandb.upload_all()
        
        # print info
        if self.cfg.env_id_print_data in env_ids and self.common_step_counter > 0:
            reward_items = ['standby_reward', 'dist_reward', 'pose_reward', 'lift_reward', 'angl_reward', 'orient_reward', 'act_penalty']
            total_reward = sum([abs(self.extras[item]) for item in reward_items])

            print("\n")
            print("#" * 17, " Statistics", "#" * 17)
            # print(f"env id:   {self.cfg.env_id_print_data}")
            print(f"subtask_index: {self._subtask_index.float().mean().item()}")
            print(f"success:  {self._successed.float().mean().item()*100.0:.2f}%")
            print(f"standby_reward:   {self.extras['standby_reward']:.2f} ({(abs(self.extras['standby_reward']) / total_reward * 100):.2f}%)")
            print(f"dist_reward:      {self.extras['dist_reward']:.2f} ({(abs(self.extras['dist_reward']) / total_reward * 100):.2f}%)")
            print(f"angle_reward:     {self.extras['angl_reward']:.2f} ({(abs(self.extras['angl_reward']) / total_reward * 100):.2f}%)")
            print(f"pose_reward:      {self.extras['pose_reward']:.2f} ({(abs(self.extras['pose_reward']) / total_reward * 100):.2f}%)")
            print(f"lego_up_reward:   {self.extras['lift_reward']:.2f} ({(abs(self.extras['lift_reward']) / total_reward * 100):.2f}%)")
            print(f"orient_reward:    {self.extras['orient_reward']:.2f} ({(abs(self.extras['orient_reward']) / total_reward * 100):.2f}%)")
            print(f"action_penalty:   {self.extras['act_penalty']:.2f} ({(abs(self.extras['act_penalty']) / total_reward * 100):.2f}%)")
            print(f"total_reward:     {sum([self.extras[item] for item in reward_items]):.2f}")
            print("#" * 15, "Statistics End", "#" * 15,"\n")
            
            self.extras = {'standby_reward': 0, 'dist_reward': 0, 'pose_reward': 0, 'lift_reward': 0, 'angl_reward': 0, 'orient_reward': 0, 'act_penalty': 0, 'success': 0}

        self._target_init_pose[env_ids, :7] = self._target.data.root_link_state_w[env_ids, :7].clone()
        self._target_init_pose[:, :3] -= self.scene.env_origins
        # compute reward
        self._compute_intermediate_values()

        self._hand_target_pose = torch.zeros_like(self._hand_init_pose)
        self._reset_task_target(env_ids)

        self._get_initial_reward()
        
    
    def _reset_task_target(self, env_ids: torch.Tensor):
        num_envs = len(env_ids)
        for key, value in self.cfg.reward_func[0].items():
            self.reward_func_active[key][env_ids] = torch.tensor(value, device=self.device).repeat(num_envs)
        self._subtask_index[env_ids] = torch.zeros(num_envs, device=self.device, dtype=torch.int32)
        self._target_lift_pose[env_ids] = torch.tensor(self.cfg.lift_targets[0], device=self.device).repeat(num_envs, 1)
        self._hand_init_pose[env_ids] = self.hand_base_state[env_ids, :3].clone()

    def _check_shift_subtask(self):
        """
        1.检查任务是否已经失败，若失败则重置任务
        2.检查当前子任务是否完成
        3.如果完成，则更新子任务，包括任务目标和reward fucntion
        """
        task_failed_env_ids = self._is_task_failed()
        if len(task_failed_env_ids) > 0:
            self._reset_task_target(task_failed_env_ids)
            self._get_initial_reward()

        self.subtask_finished = self._is_subtask_success(self._target_lift_pose[:, :3], self.object_state[:, :3])
        if torch.any(self.subtask_finished):
            # print(cp.LYH_DEBUG("subtask_finished:"), cp.LYH_DEBUG(self.subtask_finished))
            self.subtask_finished = self.subtask_finished & self._task_playing
            # print(cp.LYH_DEBUG("subtask_finished after:"), cp.LYH_DEBUG(self.subtask_finished))

            task_to_next_mask = (self._subtask_index < len(self.cfg.lift_targets) - 1) & self.subtask_finished
            # update subtask object
            self._subtask_index[task_to_next_mask] += 1
            # print(cp.LYH_DEBUG("subtask_index:"), cp.LYH_DEBUG(self._subtask_index))
            self._task_playing = self._subtask_index < len(self.cfg.lift_targets) - 1
            # print(cp.LYH_DEBUG("task_playing:"), cp.LYH_DEBUG(self._task_playing))
            # self._target_lift_pose[self._task_playing] = torch.tensor(self.cfg.lift_targets[self._subtask_index[self._task_playing]], device=self.device)

            # update lift target
            task_shift_mask = self._task_playing & task_to_next_mask
            if torch.any(task_shift_mask):
                subtask_indices = self._subtask_index[task_shift_mask]
                lift_targets = torch.tensor([self.cfg.lift_targets[idx] for idx in subtask_indices], device=self.device)
                self._target_lift_pose[task_shift_mask] = lift_targets
            
            # uodate reward function active and hand target pose
            if torch.any(task_to_next_mask):
                indices = self._subtask_index[task_to_next_mask]   
                for key, _ in self.cfg.reward_func[0].items():
                    reward_active = torch.tensor([self.cfg.reward_func[idx][key] for idx in indices], device=self.device)
                    self.reward_func_active[key][task_to_next_mask] = reward_active
                
                standby_task_mask = self.reward_func_active['standby']
                self._hand_target_pose = torch.where(
                    standby_task_mask.unsqueeze(1).expand_as(self._hand_init_pose),
                    self._hand_init_pose,
                    torch.zeros_like(self._hand_init_pose)
                )
                
                # reset pre energy
                self._get_initial_reward()

                # print(cp.green(f"env 0 subtask shift {self._subtask_index[0] - 1} to {self._subtask_index[0]}"))
            
                

    def _compute_intermediate_values(self):
        self._hand_index = self._finger_tip_index + [self._hand_base_link_index]
        (
            self.object_state,
            self.finger_thumb_state,
            self.finger_index_state,
            self.finger_middle_state,
            self.middle_point_state,
            self.hand_base_state,
            self.hand_state,
            self.dof_pos,
            self.dof_vel,
        ) = _compute_values(
            self.scene.env_origins,
            self._target.data.root_link_state_w.clone(),
            self._robot.data.body_link_state_w[:, self._hand_index].clone(),
            self._robot_index,
            self._robot.data.joint_pos,
            self._robot.data.joint_vel,
        )
    
    def _get_full_observations(self):
        ### fs version
        # self._obs = torch.cat(
        #         (
        #             # robot state
        #             unscale(self.dof_pos, self._joint_limit_lower[:, self._robot_index], self._joint_limit_upper[:, self._robot_index]),
        #             self.cfg.vel_obs_scale * self.dof_vel,
        #             # object state
        #             self.object_state[:, :7],
        #             self.cfg.vel_obs_scale * self.object_state[:, 7:],
        #             # goal
        #             # fingertips
        #             (self.hand_state[:, :, :3] - self.object_state[:, None, :3]).reshape(self.num_envs, -1),
        #             self.cfg.vel_obs_scale * self.hand_state[:, :, 3:].reshape(self.num_envs, -1),
        #             # actions
        #             self.actions,
        #         ),
        #         dim=-1,
        #     )

        ### teacher version, inlcude subtask target
        self.finger_normal_vec = quat_rotate_vector(self.hand_state[:,:3,3:7], torch.tensor([1., 0., 0.],device=self.device))
        self.fingertip_point =  self.object_state[:, :3].unsqueeze(1) - self.hand_state[:,:3,:3]
        cos_theta = (self.finger_normal_vec * self.fingertip_point).sum(-1) 
        self._obs = torch.cat(
                (
                    # robot state
                    # 关节角度13个
                    unscale(self.dof_pos, self._joint_limit_lower[:, self._robot_index], self._joint_limit_upper[:, self._robot_index]),
                    # 关节速度
                    self.cfg.vel_obs_scale * self.dof_vel,

                    # object state
                    # 物体位置、旋转7个
                    self.object_state[:, :7],
                    # 手位置 6*7=42个
                    self.hand_state[:, :, :7].reshape(self.num_envs, -1),
                    # 物体速度 7个
                    self.cfg.vel_obs_scale * self.object_state[:, 7:],

                    # goal
                    # 物体目标位置 3个
                    self._target_lift_pose[:, :3],
                    # 手目标位置 3个
                    self._hand_target_pose[:, :3],

                    # fingertips
                    # 手与目标位置差 6*3=18个
                    (self.hand_state[:, :, :3] - self.object_state[:, None, :3]).reshape(self.num_envs, -1),
                    # 手速度
                    self.cfg.vel_obs_scale * self.hand_state[:, :, 3:].reshape(self.num_envs, -1),
                    # 指尖法向量夹角cos值 3个
                    cos_theta,
                    
                    # actions
                    # 关节角度13个
                    self.actions,
                ),
                dim=-1,
            )

    def _get_initial_reward(self):
        standby_active = self.reward_func_active['standby']
        position_active = self.reward_func_active['position']
        grasp_active = self.reward_func_active['grasp']
        orientation_active = self.reward_func_active['orientation']
        standby_reward, distance_reward, pose_reward, angle_reward, lift_reward, orientation_reward, _ = _compute_rewards(
            self.finger_thumb_state,
            self.finger_index_state,
            self.finger_middle_state,
            self.middle_point_state,
            self._hand_target_pose,
            self.hand_base_state[:, :3],
            self._target_init_pose[:, :7],
            self.object_state[:, :7],
            self.actions[:, :self._arm_joint_num],
            self._curr_targets[:, self._hand_real_joint_index],
            self._prev_targets[:, self._hand_real_joint_index],
            self._z_unit_tensor,
            self._target_lift_pose,
            standby_active,
            position_active,
            grasp_active,
            orientation_active,
        )
        # print(cp.LYH_DEBUG("init target init pose:"), cp.LYH_DEBUG(euler_from_quat(self._target_init_pose[0, 3:7])))
        # print(cp.LYH_DEBUG("init object state:"), cp.LYH_DEBUG(euler_from_quat(self.object_state[0, 3:7])))
        # print(cp.LYH_DEBUG("orientation_reward_init:"), cp.LYH_DEBUG(orientation_reward[0]))
        # orientation_reward = torch.zeros_like(distance_reward)
        self._pre_standby_reward = standby_reward
        self._pre_distance_reward = distance_reward
        self._pre_pose_reward = pose_reward
        self._pre_angle_reward = angle_reward
        self._pre_lift_reward = lift_reward
        self._pre_orientation_reward = orientation_reward
        self._pre_energy = standby_reward + distance_reward + pose_reward + angle_reward + lift_reward + orientation_reward
    
    def _maker_visualizer(self):
        if self.cfg.enable_marker and self._visualizer is not None:
            thumb_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[0],:]
            index_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[1],:]
            middle_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[2],:]
            ring_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[3],:]
            pinky_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[4],:]
            grasp_fingers_pos = (thumb_tip_link_state[:,:3] + index_tip_link_state[:,:3]) / 2
            target_state = self._target.data.root_link_state_w[:,:]

            # refresh visualize and marker
            marker_pos = torch.cat((
                self._final_target_pose[0:1,:3], # 目标位置
                thumb_tip_link_state[0:1,:3],
                index_tip_link_state[0:1,:3],
                middle_tip_link_state[0:1,:3],
                ring_tip_link_state[0:1,:3],
                pinky_tip_link_state[0:1,:3],
                target_state[0:1,:3],
                # grasp_fingers_pos[0:1,:3]
                ),0)
            
            marker_rot = torch.cat((
                torch.tensor([[1., 0., 0., 0.]], device=self.device),
                thumb_tip_link_state[0:1,3:7],
                index_tip_link_state[0:1,3:7],
                middle_tip_link_state[0:1,3:7],
                ring_tip_link_state[0:1,3:7],
                pinky_tip_link_state[0:1,3:7],
                target_state[0:1,3:7],
                # torch.zeros((1,4),device=self.device)
                ),0)

            self._visualizer.visualize(marker_pos, marker_rot)

def quat_to_euler_xyz(q, *, degrees=True):
    """
    四元数 [x, y, z, w]  →  (roll, pitch, yaw)  (XYZ 顺序)
    默认返回角度制；将 degrees=False 可改返回弧度。
    """
    if isinstance(q, torch.Tensor):
        q = q.detach().cpu().numpy()
    else:
        q = np.asarray(q, dtype=float)
    # ---- 单位化 ----
    q = q / np.linalg.norm(q)

    x, y, z, w = q
    # 旋转矩阵分量
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    # 根据 XYZ (roll-pitch-yaw) 推导出的解析式
    pitch = np.arcsin(-2 * (xz - wy))              # ∈ [-π/2, π/2]
    roll  = np.arctan2( 2 * (yz + wx), 1 - 2*(yy + zz))
    yaw   = np.arctan2( 2 * (xy + wz), 1 - 2*(xx + zz))

    if degrees:
        return np.degrees([roll, pitch, yaw])
    return roll, pitch, yaw


def euler_from_quat(q):
    """包装 API：输入四元数，直接打印欧拉角（度）。"""
    roll, pitch, yaw = quat_to_euler_xyz(q, degrees=True)
    return f"Euler XYZ (deg) → roll: {roll:.2f},  pitch: {pitch:.2f},  yaw: {yaw:.2f}"

# helper functions
@torch.jit.script
def planar_force_closure(
    p: torch.Tensor,            # (B,3,3)  指尖位置  (x,y,z)
    n: torch.Tensor,            # (B,3,3)  指尖内法向 (单位向量)
    eps0: float = 0.05          # 缩放参数,tanh横向压缩
) -> torch.Tensor:              # (B,)      批量奖励
    """
    6d扳手力闭包的简化模型[Fx,Fy,Fz,τx,τy,τz]->[Fx,Fy,τz]
    只考虑xy平面的扰动
    """
    # ---- 1. 取 x、y 分量 ----
    n_xy = n[:, :, :2]                 # (B,3,2)  提取法向的 x、y
    px, py = p[:, :, 0], p[:, :, 1]    # (B,3)    指尖 x、y 坐标
    wx, wy = n_xy[:, :, 0], n_xy[:, :, 1]  # (B,3) 法向的 x、y 分量

    # ---- 2. 计算 z 方向力矩 τ_z = (p × n)_z ----
    tau_z = px * wy - py * wx          # (B,3)    行列式公式

    # ---- 3. 拼成平面扳手列 w_planar ----
    w_planar = torch.stack([wx, wy, tau_z], dim=-1)  # (B,3,3)

    # ---- 4. 列向量单位化 ----
    w_planar = torch.nn.functional.normalize(w_planar, dim=-1)

    # ---- 5. 变成 3×3 抓取矩阵 G ----
    G = w_planar.permute(0, 2, 1)      # (B,3,3)  shape: (B, 行=3, 列=3)

    # ---- 6. 奇异值分解 SVD ----
    _, S, _ = svd(G)                   # S: (B,3)  降序奇异值

    # ---- 7. 提取最小奇异值 σ_min ----
    sigma_min = S[:, -1]               # (B,)     每批次最小值

    # ---- 8. 平滑映射成奖励 ----
    return torch.tanh(sigma_min / eps0)

@torch.jit.script
def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    将四元数 (w,x,y,z) 转为旋转矩阵。
    参数
    ----
    q : Tensor[..., 4]

    返回
    ----
    R : Tensor[..., 3, 3]
    """
    eps: float = 1e-8
    # --- 归一化 ---
    norm = torch.norm(q, p=2, dim=-1, keepdim=True)
    q = q / torch.clamp(norm, min=eps)

    # 拆分分量（TorchScript 不支持一次性“多变量解包”）
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    # --- 预计算重复项 ---
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    two: float = 2.0  # 方便和论文公式对照

    # 每个元素都是 [...]-shaped Tensor
    r00 = 1.0 - two * (yy + zz)
    r01 = two * (xy - wz)
    r02 = two * (xz + wy)

    r10 = two * (xy + wz)
    r11 = 1.0 - two * (xx + zz)
    r12 = two * (yz - wx)

    r20 = two * (xz - wy)
    r21 = two * (yz + wx)
    r22 = 1.0 - two * (xx + yy)

    # --- 组装旋转矩阵 ---
    R = torch.stack((
            torch.stack((r00, r01, r02), dim=-1),
            torch.stack((r10, r11, r12), dim=-1),
            torch.stack((r20, r21, r22), dim=-1)
        ), dim=-2)          # 最后两维变成 3×3
    return R

@torch.jit.script
def quat_rotate_vector(q: torch.Tensor,          # [..., 4]
                       v: torch.Tensor           # [..., 3]
                      ) -> torch.Tensor:         # [..., 3]
    """
    用四元数 q 旋转任意向量 v。
    q 最后维度是 (w,x,y,z)；q 与 v 的批量维可广播。
    """
    R = quat_to_rotmat(q)                        # [..., 3, 3]
    v_expanded = v.unsqueeze(-1)                 # [..., 3, 1]
    v_rot = torch.matmul(R, v_expanded)          # [..., 3, 1]
    return v_rot.squeeze(-1)                     # [..., 3]

@torch.jit.script
def cross2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """ 2-D 向量叉积（返回标量），支持 `(…,2)` broadcast """
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

@torch.jit.script
def rotate_cw(x: torch.Tensor) -> torch.Tensor:
    # 顺时针旋转 90°： (x,y)->(y,-x)
    return torch.stack((x[:, 1], -x[:, 0]), dim=1)
    
@torch.jit.script
def compute_support_polygon(tip_xy: torch.Tensor, 
                            com_xy: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    使用拇指、食指、中指凸包support polygon计算抓取姿态的reward

    Args:
        tip_xy: 三指xy面投影, shape (batch, 3, 2)
        com_xy: 目标质心xy面投影, shape (batch, 2)
    
    Returns:
        d_signed: 质心与凸包的带符号距离, 负表示在凸包内, 正表示在凸包外, 0表示在凸包上, shape (batch,)
        inradius: 凸包内半径, shape (batch,)
        incenter: 凸包内心xy坐标, shape (batch, 2)
    """
    # 保证三指凸包顺序为逆时针，以方便求外法向量
    area = cross2d(tip_xy[:,1]-tip_xy[:,0], tip_xy[:,2]-tip_xy[:,0])
    swap = area < 0
    if swap.any():
        perm = torch.tensor([0, 2, 1], device=tip_xy.device)   # finger 0,2,1
        tip_xy[swap] = tip_xy[swap][:, perm, :]                # (…,3,2) -> (…,3,2)

    # 三条边
    v0, v1, v2 = tip_xy[:, 0, :], tip_xy[:, 1, :], tip_xy[:, 2, :]  # (B, 2)
    e0, e1, e2 = v1 - v0, v2 - v1, v0 - v2  # (B, 2)

    # 计算外法向量
    n0, n1, n2 = rotate_cw(e0), rotate_cw(e1), rotate_cw(e2)  # (B, 2)
    
    # 半空间方程中的常数项
    b0 = -(n0 * v0).sum(dim=-1)
    b1 = -(n1 * v1).sum(dim=-1)
    b2 = -(n2 * v2).sum(dim=-1)

    # 使用半空间方程计算距离，距离为负表示在凸包内，距离为正表示在凸包外，距离为0表示在凸包上
    l0 = ((n0 * com_xy).sum(dim=-1) + b0) / torch.norm(n0, p=2, dim=-1)
    l1 = ((n1 * com_xy).sum(dim=-1) + b1) / torch.norm(n1, p=2, dim=-1)
    l2 = ((n2 * com_xy).sum(dim=-1) + b2) / torch.norm(n2, p=2, dim=-1)
    d_signed = torch.max(torch.stack([l0, l1, l2], 1), 1).values  # (B,)

    # 计算凸包三角形内半径
    a = torch.norm(e1, dim=1)      # |v1-v2|  — 边长 a (对顶 v0)
    b = torch.norm(e2, dim=1)      # |v2-v0|  — 边长 b (对顶 v1)
    c = torch.norm(e0, dim=1)      # |v0-v1|  — 边长 c (对顶 v2)
    peri = a + b + c        # 周长
    incenter = (a[:, None] * v0 +
                b[:, None] * v1 +
                c[:, None] * v2) / peri[:, None]             # 内心(B,2)
    area2 = cross2d(v1 - v0, v2 - v0).abs()  # 2*Area
    inradius = area2 / peri                  # r = 2A / (a+b+c)
    
    return d_signed, inradius, incenter

@torch.jit.script
def tolerance(x: torch.Tensor, y: torch.Tensor, r: float, margin: float = 0.0, 
              value_at_margin: float = 0.1) -> torch.Tensor:
    """Returns 1 when `x` falls inside the circle centered at `y` with radius `r`"""
    if margin < 0.0:
        raise ValueError('margin must be non-negative')

    # Calculate the Euclidean distance from each point in x to y
    distance = torch.norm(x - y, p=2, dim=-1)
    
    # Calculate in_bounds mask
    in_bounds = distance <= r
    
    # Handle the zero margin case
    if margin == 0.0:
        return torch.where(in_bounds, torch.ones_like(distance), torch.zeros_like(distance))
    
    # Calculate normalized distance for sigmoid
    d = (distance - r) / margin
    
    # Calculate sigmoid value for out-of-bounds points
    scale = torch.sqrt(-2.0 * torch.log(torch.tensor(value_at_margin, device=x.device)))
    sigmoid_value = torch.exp(-0.5 * (d * scale) ** 2)
    
    # Combine in_bounds and out_of_bounds values
    return torch.where(in_bounds, torch.ones_like(distance), sigmoid_value)

@torch.jit.script
def compute_angle_line_plane(p1: torch.Tensor, p2: torch.Tensor, plane_normal: torch.Tensor) -> torch.Tensor:
    """Compute angle between a line (defined by two points) and a plane (defined by its normal)
    
    Args:
        p1: First point of the line, shape (batch, 3)
        p2: Second point of the line, shape (batch, 3) 
        plane_normal: Normal vector of the plane, shape (batch, 3)
    
    Returns:
        Angle between line and plane in radians, shape (batch)
    """
    # Compute the direction vector of the line
    line_direction = p2 - p1  # (batch, 3)
    
    # Normalize the line direction and the plane normal
    line_direction_norm = torch.norm(line_direction, dim=-1, keepdim=True)
    plane_normal_norm = torch.norm(plane_normal, dim=-1, keepdim=True)
    
    # Add small epsilon to avoid division by zero
    eps = torch.tensor(1e-8, device=p1.device)
    line_direction_normalized = line_direction / (line_direction_norm + eps)
    plane_normal_normalized = plane_normal / (plane_normal_norm + eps)
    
    # Compute the dot product between the line direction and the plane normal
    dot_product = torch.sum(line_direction_normalized * plane_normal_normalized, dim=-1)
    
    # Clamp the dot product to avoid numerical issues with acos
    dot_product_clamped = torch.clamp(dot_product, -1.0 + eps, 1.0 - eps)
    
    # Compute the angle between the line direction and the plane normal
    angle_with_normal = torch.acos(dot_product_clamped)
    
    # Compute the angle between the line and the plane
    angle_line_plane = torch.tensor(torch.pi/2, device=p1.device) - angle_with_normal
    
    return angle_line_plane

@torch.jit.script
def _quat_sin2_loss(a: torch.Tensor, b: torch.Tensor):
    """
    返回 sin²(θ/2)，数值域 0~1, 和角度成单调关系, θ 为 a 和 b 之间的夹角
    
    Args:
        a: 第一个四元数张量
        b: 第二个四元数张量
        
    Returns:
        torch.Tensor: sin²(θ/2) 的值
    """
    # 确保输入是四元数
    assert a.shape[-1] == 4 and b.shape[-1] == 4, "输入必须是四元数"
    
    # 归一化四元数
    a_norm = a / torch.norm(a, dim=-1, keepdim=True)
    b_norm = b / torch.norm(b, dim=-1, keepdim=True)
    
    # 计算点积
    dot = torch.sum(a_norm * b_norm, dim=-1)
    
    # 计算 sin²(θ/2)
    return 1.0 - dot.pow(2)

@torch.jit.script
def _compute_rewards(
    finger_thumb_state: torch.Tensor,
    finger_index_state: torch.Tensor,
    finger_middle_state: torch.Tensor,
    middle_point_state: torch.Tensor,
    hand_target_pose: torch.Tensor,
    hand_base_state: torch.Tensor,
    lego_init_state: torch.Tensor,
    lego_state: torch.Tensor,
    arm_actions: torch.Tensor,
    joint_pos_target: torch.Tensor,
    prev_joint_pos_target: torch.Tensor,
    z_unit_tensor: torch.Tensor,
    lift_target_pose: torch.Tensor,
    standby_active: torch.Tensor,
    position_active: torch.Tensor,
    grasp_active: torch.Tensor,
    orientation_active: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    lego_init_pos = lego_init_state[:, :3].clone()
    lego_pos = lego_state[:, :3].clone()
    lego_init_rot = lego_init_state[:, 3:7].clone()
    lego_rot = lego_state[:, 3:7].clone()
    
    ### standby reward
    standby_reward = torch.exp(-1.0 * torch.norm(hand_base_state - hand_target_pose, p=2, dim=-1)) * 200.0
    standby_reward = torch.where(standby_active, standby_reward, torch.zeros_like(standby_reward, dtype=standby_reward.dtype))

    ### grasp reward
    # define dist reward
    fingertip_pos = torch.stack([finger_thumb_state[:,:3], finger_index_state[:,:3], finger_middle_state[:,:3]], dim=0)
    finger_dist = torch.norm(lego_pos.unsqueeze(0) - fingertip_pos, p=2, dim=-1).sum(dim=0)
    distance_reward = torch.exp(-5.0 * torch.clamp((finger_dist - 0.05), 0, None))
    distance_reward = torch.where(grasp_active, distance_reward, torch.zeros_like(distance_reward, dtype=distance_reward.dtype))
    
    # # define force closure reward
    thumb_normal_vec = quat_rotate_vector(finger_thumb_state[:,3:7], torch.tensor([1., 0., 0.],device=finger_thumb_state.device))
    index_normal_vec = quat_rotate_vector(finger_index_state[:,3:7], torch.tensor([1., 0., 0.],device=finger_index_state.device))
    middle_normal_vec = quat_rotate_vector(finger_middle_state[:,3:7], torch.tensor([1., 0., 0.],device=finger_middle_state.device))
    normal_vec = torch.stack([thumb_normal_vec, index_normal_vec, middle_normal_vec], dim=-2)
    # fingertip_pos = torch.stack([finger_thumb_state[:,:3], finger_index_state[:,:3], finger_middle_state[:,:3]], dim=1)
    # fc_reward = planar_force_closure(fingertip_pos, normal_vec)
    # fc_reward = (torch.clamp(fc_reward, 0.0, 0.5) / 0.5)

    # degine finger pointing reward
    fingertip_pos = torch.stack([finger_thumb_state[:,:3], finger_index_state[:,:3], finger_middle_state[:,:3]], dim=1)
    fingertip_point = lego_pos.unsqueeze(1) - fingertip_pos
    fingertip_point = fingertip_point / (fingertip_point.norm(p=2, dim=-1, keepdim=True) + 1e-9)
    cos_theta = (normal_vec * fingertip_point).sum(-1)  
    pointing_reward = torch.clamp(cos_theta, 0.0, 0.8) / 0.8
    pointing_reward = pointing_reward.min(dim=-1).values * 0.5

    # # define 3finger hight reward
    # mean_hight = (finger_index_state[:,2] + finger_middle_state[:,2]) / 2.0
    # delta_hight = torch.abs(mean_hight - lego_pos[:,2])
    # hight_reward = 1 - (torch.clamp(delta_hight, 0.0, 0.02) / 0.02)

    # # define pose reward
    # fingertip_pos_xy = torch.stack([finger_thumb_state[:,:2], finger_index_state[:,:2], finger_middle_state[:,:2]], dim=1)
    # d_signed, inradius, _ = compute_support_polygon(fingertip_pos_xy, lego_pos[:, :2])
    # sp_reward = torch.where(d_signed < 0., torch.abs(d_signed) / (inradius + 1e-6), 0.0)
    # sp_reward = torch.clamp(sp_reward, 0.0, 0.1) / 0.1# 距离内心距离为半个内半径以内时，reward不再增长

    # pose_reward = 5.0 * hight_reward * sp_reward + pointing_reward
    # pose_reward = torch.where(grasp_active, pose_reward, torch.zeros_like(pose_reward, dtype=pose_reward.dtype))
    
    pose_dist = tolerance(middle_point_state[:,:3], lego_pos, r=0.016, margin=0.01)
    pose_reward = pose_dist * 6.0 + pointing_reward
    pose_reward = torch.where(grasp_active, pose_reward, torch.zeros_like(pose_reward, dtype=pose_reward.dtype))

    # define angle reward
    angle_dist = compute_angle_line_plane(finger_thumb_state[:,:3], finger_index_state[:,:3], z_unit_tensor)
    angle_reward = torch.exp(-1.0 * torch.abs(angle_dist)) * 0.5
    # angle_reward = torch.exp(-1.0 * torch.abs(angle_dist)) * 5.0
    angle_reward = torch.where(grasp_active, angle_reward, torch.zeros_like(angle_reward, dtype=angle_reward.dtype))

    ### position reward
    # define lift reward
    target_pos = lift_target_pose
    init_dist = torch.norm(lego_init_pos - target_pos, p=2, dim=-1)
    goal_dist = torch.norm(lego_pos - target_pos, p=2, dim=-1)
    lift_reward = 400.0 * torch.clamp((1 - goal_dist), 0.0, None)
    lift_reward = torch.where(position_active & (~grasp_active | (pose_dist >= 1.)), lift_reward, torch.zeros_like(lift_reward, dtype=lift_reward.dtype))

    ### orientation reward
    orientation_reward =  (1 - _quat_sin2_loss(lego_init_rot, lego_rot)) * 100.0
    orientation_reward = torch.where(orientation_active, orientation_reward, torch.zeros_like(orientation_reward, dtype=orientation_reward.dtype))
    
    # define action penalty
    action_penalty = 0.001 * torch.sum(arm_actions.pow_(2), dim=-1)
    action_penalty.add_(0.001 * torch.sum(
        joint_pos_target.sub_(prev_joint_pos_target).pow_(2), 
        dim=-1
    ))
    return standby_reward, distance_reward, pose_reward, angle_reward, lift_reward, orientation_reward, action_penalty

@torch.jit.script
def _compute_values(
    env_origins: torch.Tensor,
    object_state: torch.Tensor,
    hand_state: torch.Tensor,
    robot_index: list[int],
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_envs = env_origins.shape[0]
    object_state[:, :3].sub_(env_origins)
    num_indexs = hand_state.shape[1] # (num_envs, num_indexs, 3)
    
    hand_state_offset = env_origins.repeat((1, num_indexs)).reshape(num_envs, num_indexs, 3)
    hand_state[:, :, :3].sub_(hand_state_offset)
    finger_thumb_state  = hand_state[:,0, :].clone()
    finger_index_state  = hand_state[:,1, :].clone()
    finger_middle_state = hand_state[:,2, :].clone()
    hand_base_state     = hand_state[:,5, :].clone()
    middle_point_state  = (finger_thumb_state + finger_index_state) / 2
    
    # data for robot joint
    dof_pos = joint_pos[:, robot_index].clone()
    dof_vel = joint_vel[:, robot_index].clone()
    
    return object_state, finger_thumb_state, finger_index_state, finger_middle_state, middle_point_state, hand_base_state, hand_state, dof_pos, dof_vel

# scale data to [0,1]
@torch.jit.script
def norm(x, lower, upper):
    return (x-lower)/(upper-lower)

# scale data to [lower, upper]
@torch.jit.script
def scale(x, lower, upper):
    return (0.5 * (x + 1.0) * (upper - lower) + lower)

@torch.jit.script
def unscale(x, lower, upper):
    return 2.0 * (x-lower) / (upper - lower) - 1.0

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * torch.pi, x_unit_tensor), quat_from_angle_axis(rand1 * torch.pi, y_unit_tensor)
    )

def to_torch(x: list, dtype: torch.dtype = torch.float, device: str = 'cuda:0', requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)