# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

import torch

from isaaclab.utils.math import quat_from_euler_xyz


def get_random_position(num_envs:int, pos_base:list, pos_range:list[float], device)-> torch.Tensor:
    pos_range_tensor = torch.tensor(pos_range,device=device).unsqueeze(0).repeat(num_envs,1)
    pos_base_tensor = torch.tensor(pos_base,device=device).unsqueeze(0).repeat(num_envs,1)
    delta_pos_norm = 2.0 * (torch.rand((num_envs,3),device=device) - 0.5)
    return delta_pos_norm.mul(pos_range_tensor) + pos_base_tensor 

def get_random_orientation(num_envs:int, device)-> torch.Tensor:
    # torch.rand return  [0,1]
    random_eular_angle = 3.14 * 2.0 * (torch.rand((num_envs,3),device=device) - 0.5)
    return quat_from_euler_xyz(random_eular_angle[:,0],random_eular_angle[:,1],random_eular_angle[:,2])
    