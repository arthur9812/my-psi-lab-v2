# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from __future__ import annotations

""" Common Modules  """ 
import numpy
import torch

""" Isaac Lab Modules  """ 
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.assets import (
    AssetBaseCfg,
    RigidObjectCfg,
)


""" Psi Lab Modules  """ 
from psilab import PSILAB_USD_ASSET_DIR
from psilab.configs.robots.psi_dc_01 import PSI_DC_01_CFG
from psilab.scene.sence_cfg import SceneCfg
from psilab.random.random_cfg import RandomCfg,RigidRandomCfg,MaterialRandomCfg



ROOM_SCENE_CFG = SceneCfg(
        
        num_envs = 1, 
        env_spacing=4.0, 
        replicate_physics=True,
        
        # global light
        global_light_cfg = AssetBaseCfg(
            prim_path="/World/Light", 
            spawn=sim_utils.DomeLightCfg(
                intensity=3000.0, 
                color=(0.75, 0.75, 0.75)
            )
        ),

        # local light
        local_lights_cfg={},

        # robot
        robots_cfg = {
            "robot" : PSI_DC_01_CFG.replace( # type: ignore
                prim_path="/World/Robot",
                diff_ik_controllers ={})
        },
        
        # static object
        static_objects_cfg = {
            "room" : AssetBaseCfg(
                prim_path="/World/Room", 
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/envs/psi_garage_2_obj/GarageScene.usd"
                ),
                init_state = RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0), 
                    rot= (0.707, 0.707, 0.0, 0.0)
                )
            )
        },
        
        # rigid objects
        rigid_objects_cfg ={

            "table" : RigidObjectCfg(
                    prim_path="/World/Table", 
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=PSILAB_USD_ASSET_DIR + "/others/table_1157.usd",
                        # scale=(1.0, 1.0, 1.8),
                        scale=(1.0, 1.0, 2.0),

                        visual_material=None,
                        rigid_props=RigidBodyPropertiesCfg(
                            kinematic_enabled = True,
                            solver_position_iteration_count=255
                        )
                    ),
                    init_state = RigidObjectCfg.InitialStateCfg(
                        pos=(0.15, 0.0, 0.0), 
                        rot= (1.0, 0.0, 0.0, 0.0)
                    )
                ),
            "bottle" : RigidObjectCfg(
                prim_path="/World/Bottle",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/drink-B36-V1/B36.usd",
                    scale=(0.0006, 0.0006, 0.0006),
                    # usd_path=PSILAB_USD_ASSET_DIR + "/others/cube.usd",
                    # scale=(0.01,0.01,0.01),
                    visual_material=None,
                    rigid_props=RigidBodyPropertiesCfg(
                            kinematic_enabled = False,
                            solver_position_iteration_count=255
                        )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0,-0.05,0.9),
                    # pos=(0.0,0.0,0.9),

                    rot= (0.707, 0.707, 0.0, 0.0)
                    # rot= (1.0,0.0,0.0,0.0)

                )
            ),
          
        },
        
        # rigid objects
        deformable_objects_cfg ={},
        
        # camera sensor
        cameras_cfg={
            # "eye_left": CameraCfg(
            #     height=720,
            #     width=1280,
            #     data_types=['rgb'],
            #     prim_path = "/World/CameraLeft",
            #     spawn=sim_utils.PinholeCameraCfg(lock_camera=False),
            #     offset = CameraCfg().OffsetCfg(
            #         pos = (-0.3,0.033,1.6),
            #         rot = (1,0,0,0),
            #         convention='world')
            # ),
            # "eye_right": CameraCfg(
            #     height=720,
            #     width=1280,
            #     data_types=['rgb'],
            #     prim_path = "/World/CameraRight",
            #     spawn=sim_utils.PinholeCameraCfg(lock_camera=False),
            #     offset = CameraCfg().OffsetCfg(
            #         pos = (-0.3,-0.033,1.6),
            #         rot = (1,0,0,0),
            #         convention='world')
            # ),
        },
        
        # contact sensor
        contact_sensors_cfg={
            "left_hand": ContactSensorCfg(
                prim_path="/World/Robot/InspireHand_OY_Left/hand1_link_.*",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=[],
            ),
            "right_hand": ContactSensorCfg(
                prim_path="/World/Robot/InspireHand_OY_Right/hand2_link_.*",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=[],
            ),
        },

        # debug marker
        marker_cfg = None,
        # 
        random = RandomCfg(
            global_light_cfg = None,
            local_lights_cfg = None,
            rigid_objects_cfg = {
                "bottle": RigidRandomCfg(
                    random_type="range",
                    random_position=True,
                    random_orientation=False,
                    random_material=False,
                    position_range=[0.1,0.1,0.0],
                    position_list=[
                        [0.1,0.0,0.0],
                        [0.0,0.1,0.0],
                        [-0.1,0.0,0.0],
                        [0.0,-0.1,0.0],
                    ],
                    orientation_list=[
                        [0.707, 0.707, 0.0, 0.0],
                        [0.707, 0.0, 0.707, 0.0],
                        [0.707, 0.0, 0.0, 0.707]
                    ],
                    material_cfg = MaterialRandomCfg(
                        enable_random= False,
                        shader_path="/World/Bottle/Looks/material_0/material_0",
                        random_type="range",
                        material_type = "colored_texture",
                        color_range=[
                            [0,0,0],
                            [255,255,255]
                        ], # type: ignore
                        color_list = [
                            [0,32,54],
                            [231,65,0],
                            [21,123,10],
                        ], # type: ignore
                        texture_list =[
                            "/home/admin01/Work/02-PsiLab/psi-lab-v2/assets/usd/others/drink-B36-V1/textures/B36.jpg",
                            "/home/admin01/下载/20250311-092148.jpg",
                            "/home/admin01/下载/20250311-092142.jpg",
                            "/home/admin01/下载/20250311-092135.jpg",
                        ]
                    )
                )
            },


        )

    )



