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
import os
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


from pxr import Usd, UsdGeom, Gf, Sdf

def read_usd_mesh_points(usd_path):
    """
    从USD文件中读取网格点云数据
    
    Args:
        usd_path (str): USD文件路径
        
    Returns:
        np.ndarray: 点云数据，形状为[N, 3]，如果有问题则返回空数组
    """
    print(os.path.abspath(usd_path))

    if not os.path.exists(usd_path):
        print(f"USD文件不存在: {usd_path}")
        return numpy.zeros((0, 3))
    
    try:
        # 打开USD舞台
        stage = Usd.Stage.Open(usd_path)
        if not stage:
            print(f"无法打开USD舞台: {usd_path}")
            return numpy.zeros((0, 3))
            
        # 获取所有 mesh 网格
        all_points = numpy.zeros((0, 3))
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim) 
                xformable = UsdGeom.Xformable(prim)
                xform_ops = xformable.GetOrderedXformOps()
                
                # 初始化变换参数
                scale = numpy.array([1.0, 1.0, 1.0])
                rotation = numpy.array([0.0, 0.0, 0.0])  # ZYX顺序，角度制
                translation = numpy.array([0.0, 0.0, 0.0])


                for op in xform_ops:
                    op_name = op.GetName()
                    op_type = op.GetOpType()
                    # print(f"操作符: {op_name}, 类型: {op_type}")
                    
                    if "scale" in op_name:
                        scale_value = op.Get()
                        if scale_value is not None:
                            scale = numpy.array([scale_value[0], scale_value[1], scale_value[2]])
                            # print(f"缩放: {scale}")
                    elif "rotateZYX" in op_name:
                        rotate_value = op.Get()
                        if rotate_value is not None:
                            rotation = numpy.array([rotate_value[0], rotate_value[1], rotate_value[2]])
                            # print(f"旋转(ZYX角度): {rotation}")
                    elif "translate" in op_name:
                        translate_value = op.Get()
                        if translate_value is not None:
                            translation = numpy.array([translate_value[0], translate_value[1], translate_value[2]])
                            # print(f"平移: {translation}")

                scale_matrix = numpy.array([
                    [scale[0], 0, 0, 0],
                    [0, scale[1], 0, 0],
                    [0, 0, scale[2], 0],
                    [0, 0, 0, 1]
                ])
                
                rx = numpy.radians(rotation[0])
                ry = numpy.radians(rotation[1])
                rz = numpy.radians(rotation[2])
                
                cos_z, sin_z = numpy.cos(rz), numpy.sin(rz)
                rot_z = numpy.array([
                    [cos_z, -sin_z, 0, 0],
                    [sin_z, cos_z, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                cos_y, sin_y = numpy.cos(ry), numpy.sin(ry)
                rot_y = numpy.array([
                    [cos_y, 0, sin_y, 0],
                    [0, 1, 0, 0],
                    [-sin_y, 0, cos_y, 0],
                    [0, 0, 0, 1]
                ])
                cos_x, sin_x = numpy.cos(rx), numpy.sin(rx)
                rot_x = numpy.array([
                    [1, 0, 0, 0],
                    [0, cos_x, -sin_x, 0],
                    [0, sin_x, cos_x, 0],
                    [0, 0, 0, 1]
                ])
                
                rotation_matrix = numpy.matmul(numpy.matmul(rot_z, rot_y), rot_x)
                
                translation_matrix = numpy.array([
                    [1, 0, 0, translation[0]],
                    [0, 1, 0, translation[1]],
                    [0, 0, 1, translation[2]],
                    [0, 0, 0, 1]
                ])
                
                transform_matrix = numpy.matmul(numpy.matmul(translation_matrix, rotation_matrix), scale_matrix)
                
                
                points_attr = mesh.GetPointsAttr()
                
                if points_attr:
                    # 获取点数据
                    points = points_attr.Get()
                    if points:
                        # 转换为numpy数组
                        points_np = numpy.array([(p[0], p[1], p[2]) for p in points])
                        # all_points.append(points_np)
                        transformed_points = []
                        for point in points_np:
                            # 创建齐次坐标(x,y,z,1)
                            homogeneous = numpy.array([point[0], point[1], point[2], 1.0])
                            # 应用变换
                            transformed = numpy.dot(transform_matrix, homogeneous)
                            # 转回3D坐标
                            transformed_points.append(transformed[:3])
                        all_points = numpy.concatenate((all_points, numpy.array(transformed_points)), axis=0)
        
        if all_points.shape[0] == 0:
            print(f"未在USD文件中找到点云数据: {usd_path}")
            return numpy.zeros((0, 3))
            
        # 合并所有点
        return all_points
        
    except Exception as e:
        print(f"读取USD点云数据时出错: {e}")
        return numpy.zeros((0, 3))

def farthest_point_sampling(points, target_count):
    """
    使用最远点采样(FPS)算法下采样点云到指定数量
    
    Args:
        points (np.ndarray): 原始点云，形状为[N, 3]
        target_count (int): 目标点数量
        
    Returns:
        np.ndarray: 下采样后的点云，形状为[target_count, 3]
    """
    if points.shape[0] <= target_count:
        return points  # 如果原始点数少于目标点数，直接返回
    
    N = points.shape[0]
    selected_indices = numpy.zeros(target_count, dtype=numpy.int32)
    
    # 随机选择第一个点
    selected_indices[0] = numpy.random.randint(0, N)
    
    # 计算每个点到已选点集的最小距离
    distances = numpy.full(N, numpy.inf)
    
    # 选择剩余的点
    for i in range(1, target_count):
        # 上一个选择的点
        last_idx = selected_indices[i-1]
        
        # 计算所有点到这个点的距离
        dist_to_last = numpy.sum((points - points[last_idx])**2, axis=1)
        
        # 更新最小距离
        distances = numpy.minimum(distances, dist_to_last)
        
        # 选择距离最大的点
        selected_indices[i] = numpy.argmax(distances)
    
    return points[selected_indices]

def get_object_pointcloud(usd_paths, scale):
    """
    获取指定目标的点云数据，并下采样到 200 个点
    
    Args:
        usd_path (str): USD文件路径列表，[B]
        scale (tuple): 缩放比例
        
    Returns:
        np.ndarray: 点云数据，形状为[B, G, 3]，如果有问题则返回空数组
    """
    # 读取点云
    point_cloud = []
    for usd_path in usd_paths:
        points = read_usd_mesh_points(usd_path)
        # 应用缩放
        if scale:
            points = points * numpy.array(scale)
        points = farthest_point_sampling(points, 200)
        point_cloud.append(points)
    point_cloud = numpy.array(point_cloud) # (B, G, 3)
    # print(point_cloud.shape)
    return point_cloud

def get_obstacle_pointcloud(usd_paths, scale):
    """
    获取指定目标的点云数据，并下采样到 200 个点
    
    Args:
        usd_path (str): USD文件路径列表，[num, B]
        scale (tuple): 缩放比例
        
    Returns:
        np.ndarray: 点云数据，形状为[num, B, G, 3]，如果有问题则返回空数组
    """
    # 读取点云
    point_clouds = []
    for obj in usd_paths:
        point_cloud = []
        for usd_path in obj:
            points = read_usd_mesh_points(usd_path)
            # 应用缩放
            if scale:
                points = points * numpy.array(scale)
            points = farthest_point_sampling(points, 200)
            point_cloud.append(points)
        point_clouds.append(point_cloud)
    point_clouds = numpy.array(point_clouds) # (num, B, G, 3)
    # print(point_clouds.shape)
    return point_clouds

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
        self._visualized_points = torch.tensor([],device=self.device)
        self._arm_joint_num = 7
        self._episodes = 0

        # get instances in scene
        self._robot = self.scene.robots["robot"]
        self._target : RigidObject = None # type: ignore
        self._obstacle1 : RigidObject = None # type: ignore
        self._obstacle2 : RigidObject = None # type: ignore
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
        self._obstacle1_init_pose = torch.zeros((self.num_envs,7),device=self.device)
        self._obstacle2_init_pose = torch.zeros((self.num_envs,7),device=self.device)

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
            # 'dist_reward': torch.zeros((self.num_envs),device=self.device), 
            # 'lego_up_reward': torch.zeros((self.num_envs),device=self.device), 
            # "pose_reward": torch.zeros((self.num_envs),device=self.device),  
            # "angle_reward": torch.zeros((self.num_envs),device=self.device),  
            # 'action_penalty': torch.zeros((self.num_envs),device=self.device), 
            'target_distance_reward': torch.zeros((self.num_envs),device=self.device),
            'finger_dist_reward': torch.zeros((self.num_envs),device=self.device),
            'table_penalty': torch.zeros((self.num_envs),device=self.device),
            'close_penalty': torch.zeros((self.num_envs),device=self.device),
            'angle_reward': torch.zeros((self.num_envs),device=self.device),
            'hand_to_object_reward': torch.zeros((self.num_envs),device=self.device),
            'z_lift': torch.zeros((self.num_envs),device=self.device),  
            'xy_move': torch.zeros((self.num_envs),device=self.device), 
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
            obstacle1_state = self._obstacle1.data.root_link_state_w[:,:]
            obstacle2_state = self._obstacle2.data.root_link_state_w[:,:]

            # refresh visualize and marker
            marker_pos = torch.cat((
                thumb_tip_link_state[0:1,:3],
                index_tip_link_state[0:1,:3],
                middle_tip_link_state[0:1,:3],
                ring_tip_link_state[0:1,:3],
                pinky_tip_link_state[0:1,:3],
                lego_state[0:1,:3],
                grasp_fingers_pos[0:1,:3],
                self._visualized_points
                ),0)
            
            marker_rot = torch.cat((
                thumb_tip_link_state[0:1,3:7],
                index_tip_link_state[0:1,3:7],
                middle_tip_link_state[0:1,3:7],
                ring_tip_link_state[0:1,3:7],
                pinky_tip_link_state[0:1,3:7],
                lego_state[0:1,3:7],
                torch.zeros((1,4),device=self.device),
                torch.zeros((10,4),device=self.device)
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
        # print(self._hand_real_joint_index)
        # self._robot.data.joint_pos[self._hand_real_joint_index]
        joint_pos = self._robot.data.joint_pos[:,obs_joint_index].clone()
        joint_vel = self._robot.data.joint_vel[:,obs_joint_index].clone()
        # print(joint_pos.shape)
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
        obstacle1_state = self._obstacle1.data.root_link_state_w[:,:].clone()
        obstacle2_state = self._obstacle2.data.root_link_state_w[:,:].clone()

        # 转换为Local坐标系
        thumb_tip_link_state[:,:3] -= self.scene.env_origins[:,:]
        index_tip_link_state[:,:3] -= self.scene.env_origins[:,:]
        middle_tip_link_state[:,:3] -= self.scene.env_origins[:,:]
        ring_tip_link_state[:,:3] -= self.scene.env_origins[:,:]
        pinky_tip_link_state[:,:3] -= self.scene.env_origins[:,:]
        hand_base_state[:,:3] -= self.scene.env_origins[:,:]
        lego_state[:,:3] -= self.scene.env_origins[:,:]
        obstacle1_state[:,:3] -= self.scene.env_origins[:,:]
        obstacle2_state[:,:3] -= self.scene.env_origins[:,:]


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

        # critic obs不确定是什么且为和必须，缺少会报错，暂时设置为和obs一样
        observations = {"policy": self._obs, "critic":self._obs}

        return observations

    def get_usd_path(self, object: RigidObject):
        import isaacsim.core.utils.stage as stage_utils
        from isaaclab.sim.utils import find_matching_prims
        # acquire stage
        stage = stage_utils.get_current_stage()
        prims = find_matching_prims(object.cfg.prim_path)
        prim_usd_paths = []
        for prim in prims:
            prim_usd_paths.append(prim.GetPrimStack()[1].layer.identifier)
        return prim_usd_paths

    def get_list_usd_path(self, objects: list[RigidObject]):
        import isaacsim.core.utils.stage as stage_utils
        from isaaclab.sim.utils import find_matching_prims
        # acquire stage
        prim_usd_pathss = []
        stage = stage_utils.get_current_stage()
        for object in objects:
            prims = find_matching_prims(object.cfg.prim_path)
            prim_usd_paths = []
            for prim in prims:
                prim_usd_paths.append(prim.GetPrimStack()[1].layer.identifier)
            prim_usd_pathss.append(prim_usd_paths)
        return prim_usd_pathss
    
    def compute_distance_features(self, object_vertices, hand_link_positions):
        """
        计算手指链接与物体顶点之间的距离特征
        
        参数:
        object_vertices: 物体顶点坐标 (B, G, 3)，其中 B 是批次大小，G 是顶点数量
        hand_link_positions: 手指链接位置 (B, L, 3)，其中 L 是链接数量
        
        返回:
        difference_features: 手指链接到最近物体顶点的距离 (B, L)
        closest_vertices: 最近的物体顶点坐标 (B, L, 3)
        """
        
        # 扩展维度以进行广播
        hand_link_positions_expanded = hand_link_positions.unsqueeze(2)  # (B, L, 1, 3)
        object_vertices_expanded = object_vertices.unsqueeze(1)  # (B, 1, G, 3)
        
        # 计算每个 hand link 到每个物体顶点的距离
        distances = torch.norm(hand_link_positions_expanded - object_vertices_expanded, p=2, dim=-1)  # (B, L, G)
        # 查找最近的物体顶点
        min_distances, min_indices = torch.min(distances, dim=-1)  # (B, L)
        
        # 获取最小距离对应的顶点坐标
        closest_vertices = torch.gather(object_vertices, 1, min_indices.unsqueeze(-1).expand(-1, -1, 3))  # (B, L, 3)
        
        # 计算差向量
        difference_vectors = hand_link_positions - closest_vertices  # (B, L, 3)
        # 计算距离
        distance_features = torch.norm(difference_vectors, p=2, dim=-1)  # (B, L)
        
        return distance_features, closest_vertices

    def quat_conjugate(self, quat):
        """计算四元数的共轭"""
        conj = quat.clone()
        conj[..., 1:4] = -conj[..., 1:4]
        return conj

    def rotate_point_by_quat(self, point, quat):
        """
        使用四元数旋转点
        
        参数:
        point: 点 (B, N, 3)
        quat: 四元数 (B, N, 4)，格式为 [qw, qx, qy, qz]
        
        返回:
        rotated_point: 旋转后的点 (B, N, 3)
        """
        # 将点扩展为纯四元数 [0, x, y, z]
        point_quat = torch.zeros_like(quat)
        point_quat[..., 1:4] = point.clone()
        
        # 计算 q * p * q^(-1)
        q_conj = quat_conjugate(quat.clone())
        rotated_quat = quat_mul(quat_mul(quat.clone(), point_quat), q_conj)
        
        # 提取旋转后的点的向量部分
        rotated_point = rotated_quat[..., 1:4]
        
        return rotated_point

    # def visualize_point_cloud(self, point_cloud, color=None):
    #     """
    #     可视化点云，用于调试
        
    #     参数:
    #     point_cloud: 点云坐标 (B, N, 3)
    #     color: 点的颜色 (可选)，默认为红色
    #     """
    #     self.marker_system = SimpleMarkerSystem(self.scene.stage)
    #     self.marker_system.clear_all_markers()
    #     if color is None:
    #         color = numpy.array([1.0, 0.0, 0.0])  # 红色
            
    #     # 只显示第一个环境的点云
    #     points = point_cloud[0].cpu().numpy()
    #     self.marker_system.visualize_points(
    #         points[:10,:], 
    #         color=color, 
    #         size=0.01,
    #         name=f"cloud_{self.common_step_counter}"
    #     )
    '''
    def _get_current_rewards_and_penalty (self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """"
            Get Rewards and Action Penalty according to current state
        """
        # ********** Get Data First **********

        # finger tip link state
        thumb_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[0],:].clone()
        index_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[1],:].clone()

        # lego state
        lego_state =  self._target.data.root_link_state_w[:,:].clone()

        # lego target position
        lego_target_pos = self._target_init_pose[:,:3].clone() + torch.tensor([0, 0, self.cfg.lift_height_target],device=self.device).repeat(self.num_envs, 1)

        # 转换为Local坐标系
        thumb_tip_link_state[:,:3] -= self.scene.env_origins[:,:]
        index_tip_link_state[:,:3] -= self.scene.env_origins[:,:]
        lego_state[:,:3] -= self.scene.env_origins[:,:]
        lego_target_pos[:,:3] -= self.scene.env_origins[:,:]
        # ********** Get Data First **********

        # 食指和中指距离目标距离和越小,奖励越大， 最小距离 0.01
        """
            dis_reward = 1.0 * e ^ ( - 5 * x ) , x范围 [0.02,)
            dis_ward_max is 0.904837418
        """
        fingertip_pos = [thumb_tip_link_state[:,:3],index_tip_link_state[:,:3]]
        # 定义拇指和食指距离乐高块的距离之和
        finger_dist = sum([torch.norm(lego_state[:,:3] - pos, p=2, dim=-1) for pos in fingertip_pos])
        distance_reward = 1.0 * torch.exp(- 5 * torch.clamp(finger_dist - torch.tensor(0.02,device = self.device), torch.tensor(0,device = self.device), None))

        # 食指和拇指中点距目标的距离
        """
            
            if 中点与目标距离小于阈值0.016:
                pose_dist = 1
            else 
                pose_dist = 0
            
            pose_dist max is 6
        """
        grasp_fingers_pos = (thumb_tip_link_state[:,:3] + index_tip_link_state[:,:3]) / 2
        # pose_dist = tolerance(grasp_fingers_pos, lego_state[:,:3], 0.016, 0.01)
        # pose_reward = pose_dist * 6

        grasp_fingers_dis = torch.norm(lego_state[:,:3] - grasp_fingers_pos, p=2, dim=-1)
        pose_dist = 1.0 * torch.exp(- 5 * grasp_fingers_dis)
        pose_reward = pose_dist * 6
        # grasp_fingers_pos_finish = time.time()
        # print(grasp_fingers_dis)


        # define angle reward
        # Todo:确认，由拇指指尖指向食指指尖的向量和XY平面之间的夹角
        """
                               0.5 * dis_reward  
            angle_reward =  ---------------------
                               e^(abs(angle))
            angle_reward max is 0.525635548                 
        """
        angle_dist = compute_angle_line_plane(thumb_tip_link_state[:,:3], index_tip_link_state[:,:3], self._z_unit_tensor)
        angle_reward = distance_reward * torch.exp(-1.0 * torch.abs(angle_dist)) * 0.5 # type: ignore
        
        # define lift reward
        """
            lift_reward max 720
        """
        # 此处 Target pos 应该转为局部坐标系
        goal_dist = torch.norm(lego_target_pos- lego_state[:,:3], p=2, dim=-1)
        # Todo:确认lego距离期望位置的距离，与期望捡起高度的插值，有什么具体意义？是否正确
        lift_reward = pose_dist * 400 * torch.clamp((self._lift_height_target- goal_dist), -0.05, None)
        
        # Penalize actions
        action_penalty = 0.001 * torch.sum(self.actions[:, :self._arm_joint_num] ** 2, dim=-1)
        action_penalty += 0.001 * torch.sum((self._joint_pos_target[:, self._arm_joint_num:]- self._joint_pos_target_lasttime[:, self._arm_joint_num:]) ** 2, dim=-1)

        return distance_reward, pose_reward, lift_reward,angle_reward, action_penalty
    '''
    def _get_distance_features(self, finger_positions, object_state, point_cloud, visualize=False):
        point_cloud_local_batch = torch.tensor(point_cloud, device=self.device) # (B, G, 3)
        B, G = point_cloud_local_batch.shape[0], point_cloud_local_batch.shape[1]
        # print(target_point_cloud)
    
        pos = object_state[:, :3]  # (B, 3) - 已在环境局部坐标系中
        quat = object_state[:, 3:7]  # (B, 4)
        
        quat_expanded = quat.unsqueeze(1).repeat(1, G, 1)  # (B, G, 4)
        
        # 应用旋转
        point_cloud_rotated = self.rotate_point_by_quat(point_cloud_local_batch, quat_expanded)  # (B, G, 3)
        
        # 添加平移 - 位置已在环境局部坐标系中
        pos_expanded = pos.unsqueeze(1).repeat(1, G, 1)  # (B, G, 3)
        point_cloud_env = point_cloud_rotated + pos_expanded  # (B, G, 3)
        # # 可视化点云
        self._visualized_points = torch.tensor((point_cloud_env + self.scene.env_origins.unsqueeze(1).repeat(1, G, 1))[1,:10,:])

        distance_features, closest_vertices = self.compute_distance_features(
            point_cloud_env,  # 已在环境局部坐标系中的点云
            finger_positions  # 已在环境局部坐标系中的手指位置
        )
        return distance_features
    
    def _get_current_rewards_and_penalty (self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """"
            Get Rewards and Action Penalty according to current state
        """
        # ********** Get Data First **********

        # finger tip link state
        thumb_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[0],:].clone()
        index_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[1],:].clone()
        middle_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[2],:].clone()
        ring_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[3],:].clone()
        pinky_tip_link_state = self._robot.data.body_link_state_w[:,self._finger_tip_index[4],:].clone()
        hand_base_state = self._robot.data.body_link_state_w[:,self._hand_base_link_index,:].clone()
        joint_positions = self._robot.data.body_link_state_w[:,self._hand_real_joint_index,:3].clone()

        # lego state
        target_state =  self._target.data.root_link_state_w[:,:].clone()
        obstacle1_state = self._obstacle1.data.root_link_state_w[:,:].clone()
        obstacle2_state = self._obstacle2.data.root_link_state_w[:,:].clone()

        # lego target position
        target_target_pos = self._target_init_pose[:,:3].clone() + torch.tensor([0, 0, self.cfg.lift_height_target],device=self.device).repeat(self.num_envs, 1)

        # 转换为Local坐标系
        thumb_tip_link_state[:,:3] -= self.scene.env_origins[:,:]
        index_tip_link_state[:,:3] -= self.scene.env_origins[:,:]
        middle_tip_link_state[:,:3] -= self.scene.env_origins[:,:]
        ring_tip_link_state[:,:3] -= self.scene.env_origins[:,:]
        pinky_tip_link_state[:,:3] -= self.scene.env_origins[:,:]
        target_state[:,:3] -= self.scene.env_origins[:,:]
        obstacle1_state[:,:3] -= self.scene.env_origins[:,:]
        obstacle2_state[:,:3] -= self.scene.env_origins[:,:]
        # print(target_state.shape)
        target_target_pos[:,:3] -= self.scene.env_origins[:,:]
        hand_base_state[:,:3] -= self.scene.env_origins[:,:]
        joint_positions -= self.scene.env_origins[:,:].unsqueeze(1).repeat(1,joint_positions.shape[1],1)

        finger_positions = torch.stack([
            thumb_tip_link_state[:,:3],
            index_tip_link_state[:,:3],
            middle_tip_link_state[:,:3],
            ring_tip_link_state[:,:3],
            pinky_tip_link_state[:,:3],
        ], dim=1) # (B, 5, 3)
        finger_positions = torch.cat([joint_positions, finger_positions], dim=1) # (B, 11, 3)
        
        target_distance_features = self._get_distance_features(finger_positions, target_state, self._target_point_cloud)
        obs1_distance_features = self._get_distance_features(finger_positions, obstacle1_state, self._obstacle1_point_cloud)
        obs2_distance_features = self._get_distance_features(finger_positions, obstacle2_state, self._obstacle2_point_cloud, visualize=True)

        # ********** Get Data First **********

        target_dist = torch.norm(target_state[:,:3] - target_target_pos[:,:3], p=2, dim=-1)
        target_distance_reward = 5 * torch.clamp(0.2 - target_dist, 0, None)

        # 取 ( 大拇指指尖 + 食指指尖 ) /2
        point = (thumb_tip_link_state[:,:3] + index_tip_link_state[:,:3]) / 2
        dist = torch.norm(point - target_state[:,:3], p=2, dim=-1)
        dist = torch.clamp(dist - 0.01, 0, None)
        finger_dist_reward = torch.exp(-10 * dist)

        # 桌面惩罚：手的目标位置不能低于桌面。
        goal_table_dist = (target_state[:,2] - 0.7)
        # # 五根手指到桌面的高度差最小值，不带 abs
        # fact_table_dist = torch.min(torch.stack([
        #     thumb_tip_link_state[:, 2] - 0.7,
        #     index_tip_link_state[:, 2] - 0.7,
        #     middle_tip_link_state[:, 2] - 0.7,
        #     ring_tip_link_state[:, 2] - 0.7,
        #     pinky_tip_link_state[:, 2] - 0.7
        # ], dim=1), dim=1)[0]
        # table_penalty = torch.where((fact_table_dist <= 0) | (goal_table_dist <= 0), -5, 0)
        table_penalty = torch.where(goal_table_dist <= 0, -5, 0)
        # table_penalty = 0

        # 闭合惩罚：如果手在离物体较远的地方就将手闭合，则进行惩罚
        thumb_index_dist = torch.norm(
            thumb_tip_link_state[:, :3] - index_tip_link_state[:, :3],
            p=2, dim=-1
        )
        close_thresh = 0.015  
        dist_thresh = 0.02   
        penalty_value = -1 
        close_penalty = torch.where(
            (thumb_index_dist < close_thresh) & (dist > dist_thresh),
            penalty_value * torch.ones_like(thumb_index_dist),
            torch.zeros_like(thumb_index_dist)
        )
        
        # 靠近奖励：手上 11 个关键点到目标点云上最近点的平均距离 
        hand_to_object_reward = torch.mean(target_distance_features, dim=-1)  
        hand_to_object_reward = torch.clamp(hand_to_object_reward - 0.04, 0, None)
        hand_to_object_reward = torch.exp(-0.2*(hand_to_object_reward * 50))

        # 姿势奖励：由拇指指尖指向食指指尖的向量和XY平面之间的夹角 * 距离 reward
        # 大概也可以是垂直 XY ？？？
        angle_dist = compute_angle_line_plane(thumb_tip_link_state[:,:3], index_tip_link_state[:,:3], self._z_unit_tensor)
        angle_dist = torch.min(torch.stack([torch.abs(angle_dist), torch.abs(torch.pi / 2 - angle_dist)]), dim=0)[0]
        angle_reward = finger_dist_reward * torch.exp(-1.0 * torch.abs(angle_dist)) * 0.5
        
        return target_distance_reward, finger_dist_reward, table_penalty, close_penalty, hand_to_object_reward, angle_reward

    def _get_rewards(self) -> torch.Tensor:
        
        # distance_reward,pose_reward,lift_reward,angle_reward,action_penalty = self._get_current_rewards_and_penalty()

        # compute total reward
        # total_reward = (distance_reward + pose_reward + lift_reward + angle_reward  - self._reward_lasttime) - action_penalty
        
        # self.extras['dist_reward'] += distance_reward - self.pre_distance_reward # type: ignore
        # self.extras['pose_reward'] += pose_reward  - self.pre_pose_reward# type: ignore
        # self.extras['lego_up_reward'] += lift_reward - self.pre_lift_reward# type: ignore
        # self.extras['angle_reward'] += angle_reward - self.pre_angle_reward# type: ignore
        # self.extras['action_penalty'] += action_penalty # type: ignore

        target_distance_reward, finger_dist_reward, table_penalty, close_penalty, hand_to_object_reward, angle_reward = self._get_current_rewards_and_penalty()
        target_distance_reward *= 50
        hand_to_object_reward *= 2
        
        total_reward = target_distance_reward + finger_dist_reward + table_penalty + close_penalty + hand_to_object_reward + angle_reward

        self.extras['target_distance_reward'] += target_distance_reward - self.pre_target_distance_reward # type: ignore
        self.extras['finger_dist_reward'] += finger_dist_reward - self.pre_finger_dist_reward # type: ignore
        self.extras['table_penalty'] += table_penalty - self.pre_table_penalty # type: ignore
        self.extras['close_penalty'] += close_penalty - self.pre_close_penalty # type: ignore
        self.extras['hand_to_object_reward'] += hand_to_object_reward - self.pre_hand_to_object_reward # type: ignore
        self.extras['angle_reward'] += angle_reward - self.pre_angle_reward # type: ignore  
        # compute mean reward
        self.extras['mean_reward'] += total_reward.mean().to('cpu') # type: ignore

        # print(self.common_step_counter)
        # self.pre_distance_reward = distance_reward
        # self.pre_pose_reward = pose_reward
        # self.pre_lift_reward = lift_reward
        # self.pre_angle_reward = angle_reward
        # self._reward_lasttime = distance_reward + pose_reward + lift_reward + angle_reward
        self.pre_target_distance_reward = target_distance_reward
        self.pre_finger_dist_reward = finger_dist_reward
        self.pre_table_penalty = table_penalty
        self.pre_close_penalty = close_penalty  
        self.pre_hand_to_object_reward = hand_to_object_reward
        self.pre_angle_reward = angle_reward    

        # print('*******************************')
        # print('reward_total :%s毫秒' % ((reward_finish - reward_start)*1000))
        # print('distance_reward_finish :%s毫秒' % ((distance_reward_finish - reward_start)*1000))
        # print('grasp_fingers_pos_finish :%s毫秒,  %.2f' % ((grasp_fingers_pos_finish - distance_reward_finish)*1000,(grasp_fingers_pos_finish - distance_reward_finish)/(reward_finish - reward_start)))
        # print('angle_reward_finish :%s毫秒,  %.2f' % ((angle_reward_finish - grasp_fingers_pos_finish)*1000,(angle_reward_finish - grasp_fingers_pos_finish)/(reward_finish - reward_start)))
        # print('goal_dist_finish :%s毫秒,  %.2f' % ((goal_dist_finish - angle_reward_finish)*1000,(goal_dist_finish - angle_reward_finish)/(reward_finish - reward_start)))
        # print('action_penalty_finish :%s毫秒,  %.2f' % ((action_penalty_finish - goal_dist_finish)*1000,(action_penalty_finish - goal_dist_finish)/(reward_finish - reward_start)))
        # print('*******************************')
        # total_reward = torch.zeros(self.num_envs,device="cuda:0")

        return total_reward

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


        # reset_time = time.time()

        # print('*******************************')
        # print('get_dones_total :%s毫秒' % ((reset_time - get_dones_start)*1000))
        # print('time_out :%s毫秒  %.2f' % ((time_out_time - get_dones_start)*1000,(time_out_time - get_dones_start)/(reset_time - get_dones_start)))
        # print('reset_time :%s毫秒,  %.2f' % ((reset_time - time_out_time)*1000,(reset_time - time_out_time)/(reset_time - get_dones_start)))
        

        # print('*******************************')
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
        self._obstacle1 = self.scene.rigid_objects["obstacle1"]
        self._obstacle2 = self.scene.rigid_objects["obstacle2"]
        # store variables
        self._target_init_pose = self._target.data.root_link_state_w[:,:7].clone()
        self._obstacle1_init_pose = self._obstacle1.data.root_link_state_w[:,:7].clone()
        self._obstacle2_init_pose = self._obstacle2.data.root_link_state_w[:,:7].clone()
        # 这里写成了每次 reset 读一次点云，会很慢；现在每次 reset 的时候环境里的东西不会变，所以我这里特判成只有第一次 reset 时才进行读取。后面如果加入随机物品就不能这么写。
        if self._episodes == 0:
            self._target_point_cloud = get_object_pointcloud(self.get_usd_path(self._target), self._target.cfg.spawn.scale)
            self._obstacle1_point_cloud = get_object_pointcloud(self.get_usd_path(self._obstacle1), self._obstacle1.cfg.spawn.scale)
            self._obstacle2_point_cloud = get_object_pointcloud(self.get_usd_path(self._obstacle2), self._obstacle2.cfg.spawn.scale)

        # ############ Reset All Variables ################

        # fix: set pre value acoording to joint default position
        self._joint_pos_target = self._robot.data.default_joint_pos[:,self._arm_joint_index+self._hand_real_joint_index].clone()
        self._joint_pos_target_lasttime =  self._joint_pos_target.clone()

        # compute reward
        target_distance_reward, finger_dist_reward, table_penalty, close_penalty, hand_to_object_reward, angle_reward = self._get_current_rewards_and_penalty()

        # reset the pre reward
        self.pre_target_distance_reward = target_distance_reward
        self.pre_finger_dist_reward = finger_dist_reward
        self.pre_table_penalty = table_penalty
        self.pre_close_penalty = close_penalty  
        self.pre_hand_to_object_reward = hand_to_object_reward
        self.pre_angle_reward = angle_reward
        # print and log info
        if self.cfg.env_id_print_data in env_ids and self.common_step_counter > 0:
            reward_items = ['target_distance_reward', 'finger_dist_reward', 'table_penalty', 'close_penalty', 'hand_to_object_reward', 'angle_reward']
            extras = {}
            for item in reward_items:
                extras[item] = self.extras[item].to('cpu').numpy() # type: ignore
                    
            total_reward = sum([abs(extras[item][self.cfg.env_id_print_data]) for item in reward_items])

            print("\n")
            print("#" * 17, " Statistics", "#" * 17)
            print(f"env id:   {self.cfg.env_id_print_data}")
            print(f"episodes:   {self._episodes}")
            print(f"target_distance_reward:      {extras['target_distance_reward'][self.cfg.env_id_print_data]:.2f} ({(abs(extras['target_distance_reward'][self.cfg.env_id_print_data]) / total_reward * 100):.2f}%)")
            print(f"finger_dist_reward:     {extras['finger_dist_reward'][self.cfg.env_id_print_data]:.2f} ({(abs(extras['finger_dist_reward'][self.cfg.env_id_print_data]) / total_reward * 100):.2f}%)")
            print(f"table_penalty:   {extras['table_penalty'][self.cfg.env_id_print_data]:.2f} ({(abs(extras['table_penalty'][self.cfg.env_id_print_data]) / total_reward * 100):.2f}%)")
            print(f"close_penalty:   {extras['close_penalty'][self.cfg.env_id_print_data]:.2f} ({(abs(extras['close_penalty'][self.cfg.env_id_print_data]) / total_reward * 100):.2f}%)")
            print(f"hand_to_object_reward:   {extras['hand_to_object_reward'][self.cfg.env_id_print_data]:.2f} ({(abs(extras['hand_to_object_reward'][self.cfg.env_id_print_data]) / total_reward * 100):.2f}%)")
            print(f"angle_reward:   {extras['angle_reward'][self.cfg.env_id_print_data]:.2f} ({(abs(extras['angle_reward'][self.cfg.env_id_print_data]) / total_reward * 100):.2f}%)")
            print("#" * 15, "Statistics End", "#" * 15,"\n")

            # wandb log info
            if self.cfg.enable_wandb:  # type: ignore

                self._wandb.step = self._timer.run_time()
                self._wandb.set_data("performance/mean_reward",self.extras["mean_reward"]) # type: ignore
                self._wandb.upload("performance/mean_reward")


            # reset extras
            self.extras = {'target_distance_reward': 0, 'finger_dist_reward': 0, 'table_penalty': 0, 'close_penalty': 0, 'hand_to_object_reward': 0, 'angle_reward': 0, "mean_reward":0}
        
        # update episodes
        self._episodes += 1


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

    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat

def orientation_error(desired, current):
	cc = quat_conjugate(current)
	q_r = quat_mul(desired, cc)
	return q_r[:, 1:4] * torch.sign(q_r[:, 0]).unsqueeze(-1)



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