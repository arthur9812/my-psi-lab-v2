# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Common Modules  """ 
import torch

""" IsaacLab Modules  """ 
from isaaclab.assets import RigidObject
from isaaclab.sensors import ContactSensor

""" PsiLab Modules  """ 
from psilab.assets.robot_base import RobotBase


def eval_success(robot: RobotBase, target: RigidObject, contact_sensors: dict[str,ContactSensor], grasp_height:float) -> bool:
    """The evaluate of whether the grasp is successful. """

    # 成功条件 = 成功条件1 and 成功条件2
    # 成功条件1: target 在Z轴上升起的高度大于等于期望阈值
    # 成功条件2: robot双手与target的接触力数量 >= 2

    # 计算升起高度
    height_init = target.cfg.init_state.pos[2]
    height_cur = target.data.root_pos_w[0][2]
    height_lift = height_cur - height_init
    # 计算接触力数量
    contact_force_num =0
    for sensor_name,contact_sensor in contact_sensors.items():
        net_forces_w = contact_sensor.data.net_forces_w[0,:,:] # type: ignore
        for index in range(net_forces_w.size()[0]):
            if not net_forces_w[index].equal(torch.tensor([0.0,0.0,0.0],device="cuda:0")):
                contact_force_num+=1
        pass
    # print(height_cur - height_init)
    if height_lift >=grasp_height and contact_force_num>=2:
        return True

    return False


def eval_fail(robot: RobotBase, target: RigidObject, contact_sensors: dict[str,ContactSensor],has_contacted:bool) -> tuple[bool,bool]:
    """The evaluate of whether the grasp is failed. """


    # 失败条件 = 失败条件1 or 失败条件2
    # 失败条件1：target在Z轴方向上速度不为0 and target与robot没有接触点
    # 失败条件2：
    # 获取目标速度
    velocity_z = torch.round(target.data.root_link_state_w[0,9], decimals = 2)
    # 计算接触力数量
    contact_force_num = 0
    for sensor_name,contact_sensor in contact_sensors.items():
        net_forces_w = contact_sensor.data.net_forces_w[0,:,:] # type: ignore
        for index in range(net_forces_w.size()[0]):
            if not net_forces_w[index].equal(torch.tensor([0.0,0.0,0.0],device="cuda:0")):
                contact_force_num+=1
        pass

    # print(f"Velocity on Z-Axis: {velocity_z}")
    # print(f"Contact Force Num: {contact_force_num}")

    contacted = True if contact_force_num>0 else False
    has_contacted = has_contacted or contacted

    if has_contacted and velocity_z <= -0.2 and contact_force_num==0:
        return True,has_contacted
    
    return False,has_contacted