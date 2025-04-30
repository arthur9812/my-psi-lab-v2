# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Common Modules  """ 
import time
from datetime import datetime
import h5py
import json
import torch
import os

""" IsaacLab Modules  """ 
from isaaclab.utils.configclass import class_to_dict

""" Psilab Modules  """ 
from psilab.utils.h5_utils import dict_to_h5,dict_to_cpu

def create_data_buffer(env, cfg) -> dict :
    data = {}
    # add time stamps
    data["timestamps"] = []
    # add robots
    data["robots"] = {}
    for robot_name,robot in env.scene.robots.items():
        data["robots"][robot_name] = {}
        # actions
        data["robots"][robot_name]["action"] = []
        # add actuators
        for actuator_name in robot.actuators.keys():
            data["robots"][robot_name][actuator_name+"_pos"] = []
            data["robots"][robot_name][actuator_name+"_vel"] = []
        # add eef state according to ik controllers
        for ik_name in robot.ik_controllers.keys():
            data["robots"][robot_name][ik_name+"_eef_pose"] = []
            data["robots"][robot_name][ik_name+"_vel"] = []
        # add cameras 
        for camera_name,camera in robot.cameras.items():
            # multi-type
            for data_type in camera.cfg.data_types:
                data["robots"][robot_name][camera_name+ "." + data_type] = []
        # add contact sensors
        for contact_name in robot.cameras.keys():
            data["robots"][robot_name][contact_name] = []
        # extra info
        data["robots"][robot_name]["extra"] = {
            "joint_name" : {},
            "joint_index" : {},
            "cameras" : [],
        }
        # extra info: all joint
        data["robots"][robot_name]["extra"]["joint_name"]["all"] = robot.joint_names
        data["robots"][robot_name]["extra"]["joint_index"]["all"] = robot.find_joints(robot.joint_names)[0]
        for actuator_name,actuator in robot.actuators.items():
            data["robots"][robot_name]["extra"]["joint_name"][actuator_name] = actuator.joint_names
            data["robots"][robot_name]["extra"]["joint_index"][actuator_name] = actuator.joint_indices
        # extra info: cameras
        for camera_name,camera in robot.cameras.items():
            # multi-type
            for data_type in camera.cfg.data_types:
                data["robots"][robot_name]["extra"]["cameras"].append(camera_name+ "." + data_type)
    # add rigid object
    data["rigid_objects"] = {}
    for object_name in env.scene.rigid_objects.keys():
        data["rigid_objects"][object_name]=[]
    # add deformable object
    data["deformable_objects"] = {}
    for object_name in env.scene.deformable_objects.keys():
        data["deformable_objects"][object_name]=[]
    # add cameras
    data["cameras"] = {}
    for camera_name,camera in env.scene.cameras.items():
        # multi-type
        for data_type in camera.cfg.data_types:
            data["cameras"][camera_name+ "." + data_type] = []
    #
    return data

def parse_data(data: dict, env, cfg) -> dict :
    # time stamps
    data["timestamps"].append(time.time())
    # robots
    for robot_name,robot in env.scene.robots.items():
        # action
        data["robots"][robot_name]["action"].append(robot.data.joint_pos_target[0,:].clone())
        # add actuators
        for actuator_name,actuator in robot.actuators.items():
            data["robots"][robot_name][actuator_name+"_pos"].append(robot.data.joint_pos[0,actuator.joint_indices])
            data["robots"][robot_name][actuator_name+"_vel"].append(robot.data.joint_vel[0,actuator.joint_indices])
        # add eef state according to ik controllers
        for ik_name,ik in robot.ik_controllers.items():
            # transform eef position from world coordinate to robot coordinate
            eef_state = robot.data.body_link_state_w[0,ik.eef_link_index,:7].clone()
            eef_state[:3] -= robot.data.root_state_w[0,:3]
            data["robots"][robot_name][ik_name+"_eef_pose"].append(eef_state)
        # add cameras
        for camera_name,camera in robot.cameras.items():
            # multi-type
            for data_type in camera.cfg.data_types:
                image = camera.data.output[data_type].clone()
                data["robots"][robot_name][camera_name+ "." + data_type].append(image[0,:,:,:])
        # # add contact sensors
        # for contact_name,contact in robot.cameras.items():
        #     self._data["robots"][robot_name][contact_name].append()
                    
    # add rigid object
    for object_name,object in env.scene.rigid_objects.items():
        data["rigid_objects"][object_name].append(object.data.root_link_state_w[0,:7])
    # add deformable object
    # for object_name in self.scene.deformable_objects.items():
    #     self._data["deformable_objects"][object_name].append(object.data.root_link_state_w[0,:7])
    # add cameras
    for camera_name,camera in env.scene.cameras.items():
        # multi-type
        for data_type in camera.cfg.data_types:
            image = camera.data.output[data_type].clone()
            data["cameras"][camera_name+ "." + data_type].append(image[0,:,:,:])

    #
    return data

def save_data(data: dict, cfg):
    # 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/{timestamp}_data.hdf5"
    # create folder if not exist
    if not os.path.exists(cfg.output_folder):
        os.makedirs(cfg.output_folder)
    h5_file = h5py.File(cfg.output_folder+filename, 'w') # type: ignore
    # 
    aa = dict_to_cpu(data)
    dict_to_h5(aa,h5_file,"/")
    h5_file.close()
    # pass
    cfg_dict = class_to_dict(cfg.scene)
    filename = f"/{timestamp}_scene_config.json"
    json_file = open(cfg.output_folder+filename,'w') # type: ignore
    json.dump(cfg_dict,json_file,indent=4) 
    
def create_data_buffer_muilt_env(env, cfg, nums_env) -> dict :
    #
    data = {}
    for i in range(nums_env):
        data[f"env_{i}"] = {}
    
    # add time stamps
    data["timestamps"] = []
    for i in range(nums_env):
        # add robots
        data[f"env_{i}"]["robots"] = {}
        for robot_name,robot in env.scene.robots.items():
            data[f"env_{i}"]["robots"][robot_name] = {}
            # actions
            data[f"env_{i}"]["robots"][robot_name]["action"] = []
            # add actuators
            for actuator_name in robot.actuators.keys():
                data[f"env_{i}"]["robots"][robot_name][actuator_name+"_pos"] = []
                data[f"env_{i}"]["robots"][robot_name][actuator_name+"_vel"] = []
            # add eef state according to ik controllers
            for ik_name in robot.ik_controllers.keys():
                data[f"env_{i}"]["robots"][robot_name][ik_name+"_eef_pose"] = []
                data[f"env_{i}"]["robots"][robot_name][ik_name+"_vel"] = []
            # add cameras 
            for camera_name,camera in robot.cameras.items():
                # multi-type
                for data_type in camera.cfg.data_types:
                    data[f"env_{i}"]["robots"][robot_name][camera_name+ "." + data_type] = []
            # add contact sensors
            for contact_name in robot.cameras.keys():
                data[f"env_{i}"]["robots"][robot_name][contact_name] = []
            # extra info
            data[f"env_{i}"]["robots"][robot_name]["extra"] = {
                "joint_name" : {},
                "joint_index" : {},
                "cameras" : [],
            }
            # extra info: all joint
            data[f"env_{i}"]["robots"][robot_name]["extra"]["joint_name"]["all"] = robot.joint_names
            data[f"env_{i}"]["robots"][robot_name]["extra"]["joint_index"]["all"] = robot.find_joints(robot.joint_names)[0]
            for actuator_name,actuator in robot.actuators.items():
                data[f"env_{i}"]["robots"][robot_name]["extra"]["joint_name"][actuator_name] = actuator.joint_names
                data[f"env_{i}"]["robots"][robot_name]["extra"]["joint_index"][actuator_name] = actuator.joint_indices
            # extra info: cameras
            for camera_name,camera in robot.cameras.items():
                # multi-type
                for data_type in camera.cfg.data_types:
                    data[f"env_{i}"]["robots"][robot_name]["extra"]["cameras"].append(camera_name+ "." + data_type)
        # add rigid object
        data[f"env_{i}"]["rigid_objects"] = {}
        for object_name in env.scene.rigid_objects.keys():
            data[f"env_{i}"]["rigid_objects"][object_name]=[]
        # add deformable object
        data[f"env_{i}"]["deformable_objects"] = {}
        for object_name in env.scene.deformable_objects.keys():
            data[f"env_{i}"]["deformable_objects"][object_name]=[]
        # add cameras
        data[f"env_{i}"]["cameras"] = {}
        for camera_name,camera in env.scene.cameras.items():
            # multi-type
            for data_type in camera.cfg.data_types:
                data[f"env_{i}"]["cameras"][camera_name+ "." + data_type] = []
        #
    return data

def parse_data_muilt_env(data: dict, env, cfg,nums_env) -> dict :
    #
    # time stamps
    data["timestamps"].append(time.time())
    for i in range(nums_env):
        # robots
        for robot_name,robot in env.scene.robots.items():
            # action
            data[f"env_{i}"]["robots"][robot_name]["action"].append(robot.data.joint_pos_target[i,:].cpu())
            # add actuators
            for actuator_name,actuator in robot.actuators.items():
                data[f"env_{i}"]["robots"][robot_name][actuator_name+"_pos"].append(robot.data.joint_pos[i,actuator.joint_indices].cpu())
                data[f"env_{i}"]["robots"][robot_name][actuator_name+"_vel"].append(robot.data.joint_vel[i,actuator.joint_indices].cpu())
            # add eef state according to ik controllers
            for ik_name,ik in robot.ik_controllers.items():
                # transform eef position from world coordinate to robot coordinate
                eef_state = robot.data.body_link_state_w[i,ik.eef_link_index,:7].cpu()
                eef_state[:3] -= robot.data.root_state_w[i,:3]
                data[f"env_{i}"]["robots"][robot_name][ik_name+"_eef_pose"].append(eef_state.cpu())
            # add cameras
            for camera_name,camera in robot.cameras.items():
                # multi-type
                for data_type in camera.cfg.data_types:
                    image = camera.data.output[data_type].clone()
                    data[f"env_{i}"]["robots"][robot_name][camera_name+ "." + data_type].append(image[i,:,:,:].cpu())
            # # add contact sensors
            # for contact_name,contact in robot.cameras.items():
            #     self._data["robots"][robot_name][contact_name].append()
                        
        # add rigid object
        for object_name,object in env.scene.rigid_objects.items():
            data[f"env_{i}"]["rigid_objects"][object_name].append(object.data.root_link_state_w[i,:7].cpu())
        # add deformable object
        # for object_name in self.scene.deformable_objects.items():
        #     self._data["deformable_objects"][object_name].append(object.data.root_link_state_w[0,:7])
        # add cameras
        for camera_name,camera in env.scene.cameras.items():
            # multi-type
            for data_type in camera.cfg.data_types:
                image = camera.data.output[data_type].clone()
                data[f"env_{i}"]["cameras"][camera_name+ "." + data_type].append(image[i,:,:,:].cpu())

    #
    return data

def save_data_muilt_env(data: dict, cfg, nums_env:list):
    
    #
    key_pop = []
    for key in list(data.keys()):
        if len(key.split("_"))>1:
            if int(key.split("_")[-1]) not in nums_env:
                key_pop.append(key)
    
    for key in key_pop:
        data.pop(key)

    if len(list(data.keys())) <= 1:
        return      
    # 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/{timestamp}_data.hdf5"
    # create folder if not exist
    if not os.path.exists(cfg.output_folder):
        os.makedirs(cfg.output_folder)
    h5_file = h5py.File(cfg.output_folder+filename, 'w') # type: ignore
    # 
    aa = dict_to_cpu(data)
    dict_to_h5(aa,h5_file,"/")
    h5_file.close()
    # pass
    cfg_dict = class_to_dict(cfg.scene)
    filename = f"/{timestamp}_scene_config.json"
    json_file = open(cfg.output_folder+filename,'w') # type: ignore
    json.dump(cfg_dict,json_file,indent=4) 
    