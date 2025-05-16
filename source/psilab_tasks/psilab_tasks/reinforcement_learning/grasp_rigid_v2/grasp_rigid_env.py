# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Common Modules  """ 
from __future__ import annotations
import time
import torch
import numpy
import warnings
import wandb
from datetime import datetime

""" Isaac Sim Modules  """ 
import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate

""" Isaac Lab Modules  """ 
from isaaclab.sim import SimulationCfg,PhysxCfg,RenderCfg
from isaaclab.assets.rigid_object import RigidObject
from isaaclab.utils import configclass

""" Psi Lab Modules  """
from psilab import OUTPUT_DIR
from psilab.envs.rl_env import RLEnv 
from psilab.envs.rl_env_cfg import RLEnvCfg
from psilab.utils.wandb_utils import WandbLog
from psilab.utils.timer_utils import Timer

from psilab.utils.data_collect_utils import create_data_buffer,parse_data,save_data


@configclass
class GraspRigidEnvCfg(RLEnvCfg):
    """Configuration for RL environment."""

    # params
    episode_length_s = 1.0 * 210 / 60.0
    decimation = 2
    action_scale = 0.5
    action_space = 13
    observation_space = 130
    state_space = 130

    # other params from gym
    arm_hand_dof_speed_scale = 20.0
    vel_obs_scale = 0.2
    act_moving_average = 0.8
    env_id_print_data = 0 # index of env to print status
    lift_height_target = 0.3 # lego lift height target
    dist_reward_scale = 50

    # simulation config
    sim: SimulationCfg = SimulationCfg(
        dt = 1 / 120, 
        render_interval=decimation,
        physx = PhysxCfg(
            solver_type = 1, # 0: pgs, 1: tgs
            max_position_iteration_count = 16,
            max_velocity_iteration_count = 0,
            bounce_threshold_velocity = 0.002,
            enable_ccd=True,
            gpu_max_rigid_patch_count = 4096 * 4096,
            gpu_collision_stack_size = 2100000000,
            gpu_found_lost_pairs_capacity = 137401003,
            gpu_total_aggregate_pairs_capacity=5196400
        ),
        render=RenderCfg(),

    )

    # defualt ouput folder
    output_folder = OUTPUT_DIR + "/rl"

# TODO: output custom data to hdf5 files
class GraspRigidEnv(RLEnv):
    """GraspLego RL environment."""

    cfg: GraspRigidEnvCfg

    def __init__(self, cfg: GraspRigidEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        # ############### initiallize variables ###################
        self._arm_joint_num = 7
        self._episodes = 0

        # get instances in scene
        self._robot = self.scene.robots["robot"]
        self._target : RigidObject = None # type: ignore
        self._visualizer = self.scene.visualizer

        # arm joint index
        self._arm_joint_index = [self._robot.find_joints(joint_name)[0][0] for joint_name in self._robot.actuators["arm2"].joint_names]

        # hand joint index
        self._hand_joint_index = [self._robot.find_joints(joint_name)[0][0] for joint_name in self._robot.actuators["hand2"].joint_names]

        # hand base link index
        self._hand_base_link_index = self._robot.find_bodies(["hand2_link_base"])[0][0]

        # hand real joint index
        self._hand_real_joint_index = self._hand_joint_index[:6] # type: ignore
        # hand virtual joint index
        self._hand_virtual_joint_index = self._hand_joint_index[6:] # type: ignore

        # finger tip link index
        self._finger_tip_index = [
            self._robot.find_bodies(link_name)[0][0] 
            for link_name in [
                "hand2_link_1_4",
                "hand2_link_2_3",
                "hand2_link_3_3",
                "hand2_link_4_3",
                "hand2_link_5_3",
            ]]

        # joint limit
        self._joint_limit_lower = self._robot.data.joint_limits[:,:,0].clone()
        self._joint_limit_upper = self._robot.data.joint_limits[:,:,1].clone()

        # lego init pose, position and orientation(w,x,y,z)
        self._target_init_pose = torch.zeros((self.num_envs,7),device=self.device)

        # unit tensors which used to compute
        self._z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # lift height target  
        self._lift_height_target = self.cfg.lift_height_target * torch.ones(self.num_envs,device=self.device)

        # obervation state
        self._obs = torch.zeros((self.num_envs,self.cfg.observation_space),device=self.device, dtype=torch.float32) # type: ignore

        # joint target position each step, order is arm2(right), hand2(right)
        self._joint_pos_target = self._robot.data.default_joint_pos[:,self._arm_joint_index+self._hand_real_joint_index].clone()

        # joint target position for last step, include real and fake joint
        self._joint_pos_target_lasttime = self._joint_pos_target.clone()

        # reward lasttime
        self._reward_lasttime = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)

        # Print information of this environment
        print("Num envs: ", self.num_envs)
        print("Num bodies: ", self._robot.num_bodies)
        print("Num arm dofs: ", self._arm_joint_num)
        print("Num hand dofs: ", 6)
        print("hand_base_rigid_body_index: ", self._hand_base_link_index)
        
        self.extras = {
            'goal_dist_reward': torch.zeros((self.num_envs),device=self.device), 
            'hand_to_object_reward': torch.zeros((self.num_envs),device=self.device), 
            "middle_point_reward": torch.zeros((self.num_envs),device=self.device),  
            "hand_to_negative_object_reward": torch.zeros((self.num_envs),device=self.device),  
            'mean_reward': torch.zeros((1),device=self.device), 
        }
        
        # initialize wandb
        if self.cfg.enable_wandb: 
            self._wandb = WandbLog()
            project = "GraspLegoTest"
            name = "PsiLab_v2.0_RL_PPO" + datetime.strftime(datetime.now(), '%m%d_%H%M%S')
            self._wandb.init_wandb(project,name)

        # initialize Timer
        self._timer = Timer()
        
        # 用于记录物体的距离特征
        self._distance_features = torch.zeros((self.num_envs, 20, 3), device=self.device)
        self._negative_distance_features = torch.zeros((self.num_envs, 20, 3), device=self.device)
        self._middle_point = torch.zeros((self.num_envs, 3), device=self.device)
        self._last_right_hand_to_negative_object_dist = torch.ones((self.num_envs), device=self.device) * 1000
   
    def _pre_physics_step(self, actions: torch.Tensor):
        # 
        self.actions = actions.clone()
        # just for test
        # self.actions = torch.tensor([0,0.45,0,1.78,0,-0.5,-2.54,0,1.75,1.75,1.75,1.75,1.75],device="cuda:0").unsqueeze(0).repeat(self.num_envs,1)
      
    def step(self, action):
        
        # call super step first to apply action and sim step
        obs_buf,reward_buf, reset_terminated, reset_time_outs, extras = super().step(action)
        
        # refresh marker only flag is true and visualizer is valid
        if self.cfg.enable_marker and self._visualizer is not None:

            # finger tip link state
            thumb_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[0],:]
            index_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[1],:]
            middle_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[2],:]
            ring_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[3],:]
            pinky_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[4],:]
            grasp_fingers_pos = (thumb_tip_link_state[:,:3] + index_tip_link_state[:,:3]) / 2

            # lego state
            lego_state = self._target.data.root_link_state_w[:,:]

            # refresh visualize and marker
            marker_pos = torch.cat((
                thumb_tip_link_state[0:1,:3],
                index_tip_link_state[0:1,:3],
                middle_tip_link_state[0:1,:3],
                ring_tip_link_state[0:1,:3],
                pinky_tip_link_state[0:1,:3],
                lego_state[0:1,:3],
                grasp_fingers_pos[0:1,:3]
                ),0)
            
            marker_rot = torch.cat((
                thumb_tip_link_state[0:1,3:7],
                index_tip_link_state[0:1,3:7],
                middle_tip_link_state[0:1,3:7],
                ring_tip_link_state[0:1,3:7],
                pinky_tip_link_state[0:1,3:7],
                lego_state[0:1,3:7],
                torch.zeros((1,4),device=self.device)
                ),0)

            self._visualizer.visualize(
                marker_pos, 
                marker_rot)

        return obs_buf,reward_buf, reset_terminated, reset_time_outs, extras

    def _apply_action(self):
        

        # actions 范围 -1 到 1 
        # action index 0-6 : arm velocity (normed), order is same with self._arm_joint_index
        # action index 7-12 : hand real joint position target (normed), order is same with self._hand_real_joint_index

        # ============ arm ============
        # 计算arm position
        self._joint_pos_target[:, :self._arm_joint_num] = self._robot.data.joint_pos[:, self._arm_joint_index] + self.cfg.arm_hand_dof_speed_scale  * self.physics_dt * self.actions[:, :self._arm_joint_num]
        # 裁减
        self._joint_pos_target[:, :self._arm_joint_num] = tensor_clamp(
            self._joint_pos_target[:, :self._arm_joint_num], 
            self._joint_limit_lower[:,self._arm_joint_index],
            self._joint_limit_upper[:,self._arm_joint_index]
        )
        
        # ============ inspire hand ============
        # action 由 -1,1 映射到实际范围
        self._joint_pos_target[:, self._arm_joint_num:] = scale(
            self.actions[:, self._arm_joint_num:],
            self._joint_limit_lower[:,self._hand_real_joint_index],
            self._joint_limit_upper[:,self._hand_real_joint_index]
        )
     
        # 计算
        self._joint_pos_target[:, self._arm_joint_num:] = self.cfg.act_moving_average * self._joint_pos_target[:, self._arm_joint_num:] + (1.0 - self.cfg.act_moving_average) * self._joint_pos_target_lasttime[:, self._arm_joint_num:]

        # 裁减
        self._joint_pos_target[:, self._arm_joint_num:] = tensor_clamp(
            self._joint_pos_target[:, self._arm_joint_num:],
            self._joint_limit_lower[:,self._hand_real_joint_index],
            self._joint_limit_upper[:,self._hand_real_joint_index]
        )

        # store variables 
        self._joint_pos_target_lasttime = self._joint_pos_target.clone()

        # set joint position target
        self._robot.set_joint_position_target(
            self._joint_pos_target,
            self._arm_joint_index + self._hand_real_joint_index
            ) # type: ignore

    def _get_observations(self) -> dict:
        
        # ********** Get Data First **********
        # Tip： Get Link Position 都是世界坐标系，应该转换为局部坐标系

        # For test
        # if self.scene.robots["robot"].cameras:
        #     import matplotlib.pyplot as plt
        #     key = list(self.scene.robots["robot"].tiled_cameras.keys())[0]
        #     image1 = self.scene.robots["robot"].tiled_cameras[key].data.output["rgb"]
        #     image2 = self.scene.tiled_cameras["front_camera"].data.output["rgb"]
        #     image3 = self.scene.tiled_cameras["top_camera"].data.output["rgb"]
            # image = self.scene.robots["robot"].tiled_cameras[key].data.output["rgb"][1,:,:,:]
            # plt.imshow(image.cpu())
            # plt.pause(0.001)
            # parse_step_data(self._data,self,self.cfg)

        # joint index
        # obs_joint_index = torch.cat((),0) # type: ignore

        obs_joint_index = self._arm_joint_index + self._hand_real_joint_index
        # joint state
        joint_pos = self._robot.data.joint_pos[:,obs_joint_index].clone()
        joint_vel = self._robot.data.joint_vel[:,obs_joint_index].clone()

        # finger tip link state
        thumb_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[0],:].clone()
        index_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[1],:].clone() 
        middle_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[2],:].clone()
        ring_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[3],:].clone()
        pinky_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[4],:].clone()

        # hand base state 
        hand_base_state = self._robot.data.body_link_state_w[:,self._hand_base_link_index,:].clone()

        # lego state
        lego_state =  self._target.data.root_link_state_w[:,:].clone()

        # 转换为Local坐标系
        thumb_tip_link_state[:,:3] -= self.scene.env_origins[:,:]
        index_tip_link_state[:,:3] -= self.scene.env_origins[:,:]
        middle_tip_link_state[:,:3] -= self.scene.env_origins[:,:]
        ring_tip_link_state[:,:3] -= self.scene.env_origins[:,:]
        pinky_tip_link_state[:,:3] -= self.scene.env_origins[:,:]
        hand_base_state[:,:3] -= self.scene.env_origins[:,:]
        lego_state[:,:3] -= self.scene.env_origins[:,:]


        # ********** Get Observation Second **********

        # 0:12 => arm and hand joint position, 13 dim
        # tips: 归一化至 [-1,1]
        self._obs[:,:13] = unscale(
            joint_pos,
            self._joint_limit_lower[:,obs_joint_index],
            self._joint_limit_upper[:,obs_joint_index])
        

        # 13:25 => arm and hand joint velocity, 13 dim
        self._obs[:,13:26] = self.cfg.vel_obs_scale * joint_vel

        # 26:40 => 手指距离目标lego块的距离, 5 * 3 dim
        # 大拇指
        self._obs[:,26:29] =  thumb_tip_link_state[:,:3] - lego_state[:,:3]
        # 食指
        self._obs[:,29:32] =  index_tip_link_state[:,:3] - lego_state[:,:3]
        # 中指
        self._obs[:,32:35] =  middle_tip_link_state[:,:3] - lego_state[:,:3]
        # 无名指
        self._obs[:,35:38] =  ring_tip_link_state[:,:3] - lego_state[:,:3]
        # 小拇指
        self._obs[:,38:41] =  pinky_tip_link_state[:,:3] - lego_state[:,:3]

        # 41:53 => action , 13 dim
        self._obs[:,41:54] = self.actions.clone()

        # 53:59 => hand base link pose , 7 dim
        self._obs[:,54:61] = hand_base_state[:,:7]

        # 76:82 =>lego_pose, 7 dim
        self._obs[:,61:68] = lego_state[:,:7]

        # 83:88 => hand base link linear velocity and angular velocity , 6 dim
        self._obs[:,68:74] = hand_base_state[:,7:]

        # 手指指尖link的state
        # 89:138 => finger tip state: quat,linear vel,angluar vel, 10 dim
        # 大拇指，食指，中指，无名指，小拇指
        self._obs[:,74:84] = thumb_tip_link_state[:,3:]
        self._obs[:,84:94] = index_tip_link_state[:,3:]
        self._obs[:,94:104] = middle_tip_link_state[:,3:]
        self._obs[:,104:114] = ring_tip_link_state[:,3:]
        self._obs[:,114:124] = pinky_tip_link_state[:,3:]

        # 76:82 =>lego_pose, 3 dim
        self._obs[:,124:127] = lego_state[:,7:10]
        # 76:82 =>lego_pose, 3 dim
        self._obs[:,127:130] = lego_state[:,10:]

        # 计算用于奖励函数的距离特征
        # 获取手部关节位置
        hand_link_positions = torch.zeros((self.num_envs, len(self._finger_tip_index), 3), device=self.device)
        for i, link_idx in enumerate(self._finger_tip_index):
            link_state = self._robot.data.body_link_state_w[:, link_idx, :3].clone() - self.scene.env_origins[:,:]
            hand_link_positions[:, i, :] = link_state
        
        # TODO: 获取目标物体和障碍物的顶点
        # 计算手部与目标物体的距离特征
        self._distance_features, _ = compute_distance_features(target_vertices, hand_link_positions)
        
        # 计算手部与障碍物的距离特征
        self._negative_distance_features, _ = compute_distance_features(obstacle_vertices, hand_link_positions)
        
        # 计算中点 - 拇指和食指的中点
        self._middle_point = (thumb_tip_link_state[:, :3] + index_tip_link_state[:, :3]) / 2
        
        # critic obs不确定是什么且为和必须，缺少会报错，暂时设置为和obs一样
        observations = {"policy": self._obs, "critic":self._obs}

        return observations

    def _compute_hand_reward(self) -> tuple:
        """"
            Get Rewards and Action Penalty according to current state
        """
        # 获取所需状态信息
        # 目标位置 - 使用目标高度
        target_pos = self._target_init_pose[:, :3].clone() + torch.tensor([0, 0, self.cfg.lift_height_target], device=self.device).repeat(self.num_envs, 1)
        target_pos = target_pos - self.scene.env_origins[:, :]
        
        # 物体状态
        object_states = self._target.data.root_link_state_w.clone()
        object_states[:, :3] -= self.scene.env_origins[:, :]
        
        # 手部位置和旋转
        hand_base_state = self._robot.data.body_link_state_w[:, self._hand_base_link_index, :].clone()
        hand_base_state[:, :3] -= self.scene.env_origins[:, :]
        
        # 距离特征
        distance_features = self._distance_features
        negative_distance_features = self._negative_distance_features
        middle_point = self._middle_point
        last_right_hand_to_negative_object_dist = self._last_right_hand_to_negative_object_dist
        
        # 配置参数
        dist_reward_scale = self.cfg.dist_reward_scale
        
        # 计算目标距离
        goal_dist = torch.norm(target_pos - object_states[:, 0:3], p=2, dim=-1)
        
        # 目标距离奖励
        goal_dist_reward = 50 * (0.2 - goal_dist)
        
        # 手部到物体距离
        right_hand_to_object_dist = torch.sum(torch.norm(distance_features[:, :, 0:3], p=2, dim=-1, keepdim=False), dim=1) / 20
        right_hand_to_object_dist = torch.clamp(right_hand_to_object_dist - 0.04, 0, None)
        
        # 手部到负样本物体距离
        right_hand_to_negative_object_dist = torch.sum(torch.norm(negative_distance_features[:, :, 0:3], p=2, dim=-1, keepdim=False), dim=1) / 20
        right_hand_to_negative_object_dist = torch.where(
            last_right_hand_to_negative_object_dist > right_hand_to_negative_object_dist, 
            right_hand_to_negative_object_dist, 
            last_right_hand_to_negative_object_dist
        )
        
        # 中点到物体距离
        middle_point_to_object_dist = torch.norm(middle_point - object_states[:, 0:3], p=2, dim=-1)
        middle_point_to_object_dist = torch.clamp(middle_point_to_object_dist - 0.04, 0, None)
        
        # 中点奖励
        middle_point_reward = torch.exp(-0.2*(middle_point_to_object_dist * dist_reward_scale))
        
        # 手部到物体奖励
        hand_to_object_reward = torch.exp(-0.2*(right_hand_to_object_dist * dist_reward_scale))
        
        # 手部到负样本物体奖励
        hand_to_negative_object_reward = torch.exp(-0.1*(right_hand_to_negative_object_dist * dist_reward_scale)) * 1
        
        # 计算总奖励
        reward = (goal_dist_reward + hand_to_object_reward + middle_point_reward) * (1 - hand_to_negative_object_reward)
        
        # 存储奖励组件到extras中用于记录
        self.extras['goal_dist_reward'] = goal_dist_reward
        self.extras['hand_to_object_reward'] = hand_to_object_reward
        self.extras['hand_to_negative_object_reward'] = hand_to_negative_object_reward
        self.extras['middle_point_reward'] = middle_point_reward
        
        # 打印奖励信息
        if self.common_step_counter % 100 == 0:  # 每100步打印一次
            env_id = self.cfg.env_id_print_data
            print("goal_dist_reward: ", goal_dist_reward[env_id].item())
            print("hand_to_object_reward: ", hand_to_object_reward[env_id].item())
            print("hand_to_negative_object_reward: ", hand_to_negative_object_reward[env_id].item())
            print("middle_point_to_object_dist: ", middle_point_to_object_dist[env_id].item())

        return reward, right_hand_to_negative_object_dist

    def _get_rewards(self) -> torch.Tensor:
        rewards, last_right_hand_to_negative_object_dist = self._compute_hand_reward()
        
        # 更新状态
        self._last_right_hand_to_negative_object_dist = last_right_hand_to_negative_object_dist
        
        # 更新extras用于记录和显示
        self.extras['mean_reward'] += rewards.mean().to('cpu')
        
        return rewards
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        #
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # finger tip link state
        thumb_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[0],:].clone()
        index_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[1],:].clone()

        # lego state
        lego_state =  self._target.data.root_link_state_w[:,:].clone()

        # 转换为Local坐标系
        thumb_tip_link_state[:,:3] -= self.scene.env_origins[:,:]
        index_tip_link_state[:,:3] -= self.scene.env_origins[:,:]
        lego_state[:,:3] -= self.scene.env_origins[:,:]

        fingertip_pos = [thumb_tip_link_state[:,:3],index_tip_link_state[:,:3]]
        finger_dist = sum([torch.norm(lego_state[:,:3] - pos, p=2, dim=-1) for pos in fingertip_pos])

        # todo:确认重置规则，GYM源代码中只有一个env，所以不确定是一个完成所有重置，还是谁完成谁重置
        resets = torch.where(finger_dist <= -1, torch.ones(self.num_envs,device=self.device), torch.zeros(self.num_envs,device=self.device))# type: ignore

        return resets, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        
        # why do this?
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
    
        # 
        super()._reset_idx(env_ids) # type: ignore

        # run 50 step until all rigid is static
        for i in range(50):
            self.sim.step(render=True)
            self.scene.update(dt=self.physics_dt)

        # # refresh target
        # target_index = self.scene.cfg.random.task_cfg.target_indexs[0] # type: ignore
        # target_name = self.scene.cfg.random.task_cfg.target_list[target_index] # type: ignore
        self._target = self.scene.rigid_objects["target"]
        # store variables
        self._target_init_pose = self._target.data.root_link_state_w[:,:7].clone()

        # ############ Reset All Variables ################

        # fix: set pre value acoording to joint default position
        self._joint_pos_target = self._robot.data.default_joint_pos[:,self._arm_joint_index+self._hand_real_joint_index].clone()
        self._joint_pos_target_lasttime =  self._joint_pos_target.clone()

        # print and log info
        if self.cfg.env_id_print_data in env_ids and self.common_step_counter > 0:
            reward_items = ['goal_dist_reward', 'hand_to_object_reward', 'middle_point_reward', 'hand_to_negative_object_reward']
            extras = {}
            for item in reward_items:
                extras[item] = self.extras[item].to('cpu').numpy() # type: ignore
                    
            total_reward = sum([abs(extras[item][self.cfg.env_id_print_data]) for item in reward_items])

            print("\n")
            print("#" * 17, " Statistics", "#" * 17)
            print(f"env id:   {self.cfg.env_id_print_data}")
            print(f"episodes:   {self._episodes}")
            print(f"goal_dist_reward:      {extras['goal_dist_reward'][self.cfg.env_id_print_data]:.2f} ({(abs(extras['goal_dist_reward'][self.cfg.env_id_print_data]) / total_reward * 100):.2f}%)")
            print(f"hand_to_object_reward:     {extras['hand_to_object_reward'][self.cfg.env_id_print_data]:.2f} ({(abs(extras['hand_to_object_reward'][self.cfg.env_id_print_data]) / total_reward * 100):.2f}%)")
            print(f"middle_point_reward:      {extras['middle_point_reward'][self.cfg.env_id_print_data]:.2f} ({(abs(extras['middle_point_reward'][self.cfg.env_id_print_data]) / total_reward * 100):.2f}%)")
            print(f"hand_to_negative_object_reward:   {extras['hand_to_negative_object_reward'][self.cfg.env_id_print_data]:.2f} ({(abs(extras['hand_to_negative_object_reward'][self.cfg.env_id_print_data]) / total_reward * 100):.2f}%)")
            print("#" * 15, "Statistics End", "#" * 15,"\n")

            # wandb log info
            if self.cfg.enable_wandb:  # type: ignore

                self._wandb.step = self._timer.run_time()
                self._wandb.set_data("performance/mean_reward",self.extras["mean_reward"]) # type: ignore
                self._wandb.upload("performance/mean_reward")


            # reset extras
            self.extras = {'goal_dist_reward': 0, 'hand_to_object_reward': 0, 'middle_point_reward': 0, "hand_to_negative_object_reward": 0, "mean_reward":0}
        
        # update episodes
        self._episodes += 1


@torch.jit.script
def compute_distance_features(object_vertices, hand_link_positions):
    # 提取可抓握部分的顶点
    graspable_vertices = object_vertices[:, :, :]  # (B, G, 3)
    
    # 获取 batch size 和 link 数量
    B, L, _ = hand_link_positions.shape
    _, G, _ = graspable_vertices.shape
    
    # 扩展维度以进行广播
    hand_link_positions_expanded = hand_link_positions.unsqueeze(2)  # (B, L, 1, 3)
    graspable_vertices_expanded = graspable_vertices.unsqueeze(1)  # (B, 1, G, 3)
    
    # 计算每个 hand link 到每个 graspable vertex 的距离
    distances = torch.norm(hand_link_positions_expanded - graspable_vertices_expanded, dim=-1)  # (B, L, G)
    
    # 查找最近的 graspable vertex
    min_distances, min_indices = torch.min(distances, dim=-1)  # (B, L)
    
    # 获取最小距离对应的顶点坐标
    closest_vertices = torch.gather(graspable_vertices, 1, min_indices.unsqueeze(-1).expand(-1, -1, 3))  # (B, L, 3)
    
    # 计算差向量
    difference_vectors = hand_link_positions - closest_vertices  # (B, L, 3)
    
    return difference_vectors, closest_vertices

@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def compute_rewards(
    actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    up_weight: float,
    heading_weight: float,
    heading_proj: torch.Tensor,
    up_proj: torch.Tensor,
    dof_vel: torch.Tensor,
    dof_pos_scaled: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    actions_cost_scale: float,
    energy_cost_scale: float,
    dof_vel_scale: float,
    death_cost: float,
    alive_reward_scale: float,
    motor_effort_ratio: torch.Tensor,
):
    heading_weight_tensor = torch.ones_like(heading_proj) * heading_weight
    heading_reward = torch.where(heading_proj > 0.8, heading_weight_tensor, heading_weight * heading_proj / 0.8)

    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(up_proj > 0.93, up_reward + up_weight, up_reward)

    # energy penalty for movement
    actions_cost = torch.sum(actions**2, dim=-1)
    electricity_cost = torch.sum(
        torch.abs(actions * dof_vel * dof_vel_scale) * motor_effort_ratio.unsqueeze(0),
        dim=-1,
    )

    # dof at limit cost
    dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.98, dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials

    total_reward = (
        progress_reward
        + alive_reward
        + up_reward
        + heading_reward
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - dof_at_limit_cost
    )
    # adjust reward for fallen agents
    total_reward = torch.where(reset_terminated, torch.ones_like(total_reward) * death_cost, total_reward)
    return total_reward


@torch.jit.script
def compute_intermediate_values(
    targets: torch.Tensor,
    torso_position: torch.Tensor,
    torso_rotation: torch.Tensor,
    velocity: torch.Tensor,
    ang_velocity: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_lower_limits: torch.Tensor,
    dof_upper_limits: torch.Tensor,
    inv_start_rot: torch.Tensor,
    basis_vec0: torch.Tensor,
    basis_vec1: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    dt: float,
):
    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = torch_utils.maths.unscale(dof_pos, dof_lower_limits, dof_upper_limits)

    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    prev_potentials[:] = potentials
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    return (
        up_proj,
        heading_proj,
        up_vec,
        heading_vec,
        vel_loc,
        angvel_loc,
        roll,
        pitch,
        yaw,
        angle_to_target,
        dof_pos_scaled,
        prev_potentials,
        potentials,
    )

# 将数据根据上下限制归一化至 [0,1]
@torch.jit.script
def norm(x, lower, upper):
    return (x-lower)/(upper-lower)

# 将数据根据上下限制进行反归一化，由[-1,1]->[lower,upper]
@torch.jit.script
def scale(x, lower, upper):
    return (0.5 * (x + 1.0) * (upper - lower) + lower)

# 将数据根据上下限制归一化， [lower,upper]->[-1,1]
# Code Bak，
# @torch.jit.script
# def unscale(x, lower, upper):
#     return (2.0 * x - upper - lower) / (upper - lower)
# Author: FYD
@torch.jit.script
def unscale(x, lower, upper):
    return 2.0 * (x-lower) / (upper - lower) - 1.0


@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)


@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat

def orientation_error(desired, current):
	cc = quat_conjugate(current)
	q_r = quat_mul(desired, cc)
	return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)



_DEFAULT_VALUE_AT_MARGIN = 0.1

def _sigmoids(x, value_at_1, sigmoid):
    """Returns 1 when `x` == 0, between 0 and 1 otherwise.

    Args:
        x: A scalar or PyTorch tensor of shape (batch_size, 1).
        value_at_1: A float between 0 and 1 specifying the output when `x` == 1.
        sigmoid: String, choice of sigmoid type.

    Returns:
        A PyTorch tensor with values between 0.0 and 1.0.

    Raises:
        ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
          `quadratic` sigmoids which allow `value_at_1` == 0.
        ValueError: If `sigmoid` is of an unknown type.
    """
    if sigmoid in ('cosine', 'linear', 'quadratic'):
        if not 0 <= value_at_1 < 1:
            raise ValueError('`value_at_1` must be nonnegative and smaller than 1, '
                             'got {}.'.format(value_at_1))
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError('`value_at_1` must be strictly between 0 and 1, '
                             'got {}.'.format(value_at_1))

    if sigmoid == 'gaussian':
        scale = torch.sqrt(-2 * torch.log(torch.tensor(value_at_1)))
        return torch.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == 'hyperbolic':
        scale = torch.acosh(1 / torch.tensor(value_at_1))
        return 1 / torch.cosh(x * scale)

    elif sigmoid == 'long_tail':
        scale = torch.sqrt(1 / torch.tensor(value_at_1) - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == 'reciprocal':
        scale = 1 / torch.tensor(value_at_1) - 1
        return 1 / (torch.abs(x) * scale + 1)

    elif sigmoid == 'cosine':
        scale = torch.acos(2 * torch.tensor(value_at_1) - 1) / torch.pi
        scaled_x = x * scale
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore', message='invalid value encountered in cos')
            cos_pi_scaled_x = torch.cos(torch.pi * scaled_x)
        return torch.where(torch.abs(scaled_x) < 1, (1 + cos_pi_scaled_x) / 2, torch.tensor(0.0))

    elif sigmoid == 'linear':
        scale = 1 - torch.tensor(value_at_1)
        scaled_x = x * scale
        return torch.where(torch.abs(scaled_x) < 1, 1 - scaled_x, torch.tensor(0.0))

    elif sigmoid == 'quadratic':
        scale = torch.sqrt(1 - torch.tensor(value_at_1))
        scaled_x = x * scale
        return torch.where(torch.abs(scaled_x) < 1, 1 - scaled_x ** 2, torch.tensor(0.0))

    elif sigmoid == 'tanh_squared':
        scale = torch.atanh(torch.sqrt(1 - torch.tensor(value_at_1)))
        return 1 - torch.tanh(x * scale) ** 2

    else:
        raise ValueError('Unknown sigmoid type {!r}.'.format(sigmoid))

def tolerance(x, y, r, margin=0.0, sigmoid='gaussian', value_at_margin=_DEFAULT_VALUE_AT_MARGIN):
    """Returns 1 when `x` falls inside the circle centered at `y` with radius `r`, between 0 and 1 otherwise.

    Args:
        x: A batch_size x 3 numpy array representing the points to check.
        y: A length-3 numpy array representing the center of the circle.
        r: Float. The radius of the circle.
        margin: Float. Parameter that controls how steeply the output decreases as `x` moves out-of-bounds.
        sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian', 'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
        value_at_margin: A float between 0 and 1 specifying the output value when the distance from `x` to the nearest bound is equal to `margin`. Ignored if `margin == 0`.

    Returns:
        A numpy array with values between 0.0 and 1.0 for each point in the batch.

    Raises:
        ValueError: If `margin` is negative.
    """
    if margin < 0:
        raise ValueError('`margin` must be non-negative.')

    # Calculate the Euclidean distance from each point in x to p
    distance = torch.norm(x - y, p=2, dim=-1)

    in_bounds = distance <= r
    if margin == 0:
        value = torch.where(in_bounds, 1.0, 0.0)
    else:
        d = (distance - r) / margin
        
        value = torch.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

    return value

def compute_angle_line_plane(p1, p2, plane_normal):
    # Compute the direction vector of the line
    line_direction = p2 - p1  # (batch, 3)
    
    # Normalize the line direction and the plane normal
    line_direction_normalized = line_direction / torch.norm(line_direction, dim=-1, keepdim=True)  # (batch, 3)
    plane_normal_normalized = plane_normal / torch.norm(plane_normal, dim=-1, keepdim=True)  # (batch, 3)
    
    # Compute the dot product between the line direction and the plane normal
    dot_product = torch.bmm(line_direction_normalized.unsqueeze(1), plane_normal_normalized.unsqueeze(2)).squeeze()  # (batch)
    
    # Clamp the dot product to avoid numerical issues with acos
    dot_product_clamped = torch.clamp(dot_product, -1.0, 1.0)
    
    # Compute the angle between the line direction and the plane normal
    angle_with_normal = torch.acos(dot_product_clamped)  # (batch)
    
    # Compute the angle between the line and the plane
    angle_line_plane = torch.pi / 2 - angle_with_normal  # (batch)
    
    return angle_line_plane