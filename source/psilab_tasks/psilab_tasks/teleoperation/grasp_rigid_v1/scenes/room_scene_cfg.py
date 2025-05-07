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
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.camera import CameraCfg,TiledCameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.assets import (
    AssetBaseCfg,
    RigidObjectCfg,
)


""" Psi Lab Modules  """ 
from psilab import PSILAB_USD_ASSET_DIR,PSILAB_TEXTURE_ASSET_DIR
from psilab.configs.robots.psi_dc_01 import PSI_DC_01_CFG
from psilab.scene.sence_cfg import SceneCfg
from psilab.assets.robot_base_cfg import RobotBaseCfg
from psilab.controllers.differential_ik_cfg import DiffIKControllerCfg
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
        local_lights_cfg={
            # TODO: add light api
            # "Rect_Lights" : AssetBaseCfg(
            #     prim_path="/World/Lights/RectLight_.*", 
            #     spawn=None
            # )
        },

        # robot
        robots_cfg = {
            "robot" : RobotBaseCfg(
                prim_path = "/World/Robot",
                spawn = sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR+"/robots/PsiRobot_DC_01/Version_3.0/PsiRobot_DC_01_Tuned.usd",
                    activate_contact_sensors = True,

                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=True,
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=255,
                    )
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(-0.5, 0.0, 0.0),
                    rot=(1.0,0.0,0.0,0.0),
                    joint_pos={
                        "arm1_joint_link1": -0.24,
                        "arm1_joint_link2": -0.64,
                        "arm1_joint_link3": -1.52,
                        "arm1_joint_link4": -0.81,
                        "arm1_joint_link5": 0.30,
                        "arm1_joint_link6": -1.03,
                        "arm1_joint_link7": 1.35,
                        "arm2_joint_link1": 0.24,
                        "arm2_joint_link2": -0.64,
                        "arm2_joint_link3": 1.52,
                        "arm2_joint_link4": -0.81,
                        "arm2_joint_link5": -0.30,
                        "arm2_joint_link6": -1.03,
                        "arm2_joint_link7": -0.36,
                        "hand1_joint_link_1_1":0.0,
                        "hand1_joint_link_1_2":0.63,
                        "hand1_joint_link_1_3":0.03,
                        "hand1_joint_link_2_1":3.10,
                        "hand1_joint_link_2_2":1.56,
                        "hand1_joint_link_3_1":3.06,
                        "hand1_joint_link_3_2":1.56,
                        "hand1_joint_link_4_1":3.06,
                        "hand1_joint_link_4_2":1.56,
                        "hand1_joint_link_5_1":3.04,
                        "hand1_joint_link_5_2":1.56,
                        "hand2_joint_link_1_1":0.0,
                        "hand2_joint_link_1_2":0.64,
                        "hand2_joint_link_1_3":0.03,
                        "hand2_joint_link_2_1":3.11,
                        "hand2_joint_link_2_2":1.56,
                        "hand2_joint_link_3_1":3.06,
                        "hand2_joint_link_3_2":1.56,
                        "hand2_joint_link_4_1":3.08,
                        "hand2_joint_link_4_2":1.56,
                        "hand2_joint_link_5_1":3.05,
                        "hand2_joint_link_5_2":1.56,
                    }
                ),
                                    
                actuators={
                    "arm1": ImplicitActuatorCfg(
                        joint_names_expr=[
                            "arm1_joint_link1",
                            "arm1_joint_link2",
                            "arm1_joint_link3",
                            "arm1_joint_link4",
                            "arm1_joint_link5",
                            "arm1_joint_link6",
                            "arm1_joint_link7",
                            ],
                        stiffness=None,
                        damping=None,

                    ),
                    "arm2": ImplicitActuatorCfg(
                        joint_names_expr=[
                            "arm2_joint_link1",
                            "arm2_joint_link2",
                            "arm2_joint_link3",
                            "arm2_joint_link4",
                            "arm2_joint_link5",
                            "arm2_joint_link6",
                            "arm2_joint_link7",
                            ],
                        stiffness=None,
                        damping=None,
                    ),
                    "hand1": ImplicitActuatorCfg(
                        joint_names_expr=[
                            "hand1_joint_link_1_1",
                            "hand1_joint_link_1_2",
                            "hand1_joint_link_1_3",
                            "hand1_joint_link_2_1",
                            "hand1_joint_link_2_2",
                            "hand1_joint_link_3_1",
                            "hand1_joint_link_3_2",
                            "hand1_joint_link_4_1",
                            "hand1_joint_link_4_2",
                            "hand1_joint_link_5_1",
                            "hand1_joint_link_5_2"],
                        stiffness=None,
                        damping=None,

                    ),
                    "hand2": ImplicitActuatorCfg(
                        joint_names_expr=[
                            "hand2_joint_link_1_1",
                            "hand2_joint_link_1_2",
                            "hand2_joint_link_1_3",
                            "hand2_joint_link_2_1",
                            "hand2_joint_link_2_2",
                            "hand2_joint_link_3_1",
                            "hand2_joint_link_3_2",
                            "hand2_joint_link_4_1",
                            "hand2_joint_link_4_2",
                            "hand2_joint_link_5_1",
                            "hand2_joint_link_5_2"],
                        stiffness=None,
                        damping=None,

                    ),
                },
                diff_ik_controllers = {
                    "arm1":DiffIKControllerCfg(
                        command_type="pose", 
                        use_relative_mode=False, 
                        ik_method="dls",
                        joint_name=[
                            "arm1_joint_link1",
                            "arm1_joint_link2",
                            "arm1_joint_link3",
                            "arm1_joint_link4",
                            "arm1_joint_link5",
                            "arm1_joint_link6",
                            "arm1_joint_link7"
                        ],
                        eef_link_name="arm1_link7"
                    ),
                    "arm2":DiffIKControllerCfg(
                        command_type="pose", 
                        use_relative_mode=False, 
                        ik_method="dls",
                        joint_name=[
                            "arm2_joint_link1",
                            "arm2_joint_link2",
                            "arm2_joint_link3",
                            "arm2_joint_link4",
                            "arm2_joint_link5",
                            "arm2_joint_link6",
                            "arm2_joint_link7"
                        ],
                        eef_link_name="arm2_link7"
                    ),
    
                },
                cameras = {},
                tiled_cameras={
                    "base_camera": TiledCameraCfg(
                        prim_path="/World/Robot/base_camera_rgb/base_camera_rgb",
                        data_types=["rgb"],
                        width=224,
                        height=224,
                        spawn=None,
                    ),
                    "arm1_camera": TiledCameraCfg(
                        prim_path="/World/Robot/arm1_camera_rgb/arm1_camera_rgb",
                        data_types=["rgb"],
                        width=224,
                        height=224,
                        spawn=None,
                    ),
                    "arm2_camera": TiledCameraCfg(
                        prim_path="/World/Robot/arm2_camera_rgb/arm2_camera_rgb",
                        data_types=["rgb"],
                        width=224,
                        height=224,
                        spawn=None,
                    ),
                }
            )
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
                        usd_path=PSILAB_USD_ASSET_DIR + "/rigid_objects/table/table_1157.usd",
                        scale=(1.0, 1.0, 1.8),
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
                    usd_path=PSILAB_USD_ASSET_DIR + "/rigid_objects/drink-B36-V1/B36.usd",
                    scale=(0.0006, 0.0006, 0.0006),
                    visual_material=None,
                    rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=255
                        )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0,0.0,0.85),
                    rot= (0.707, 0.707, 0.0, 0.0)
                )
            ),
          
        },
        
        # rigid objects
        deformable_objects_cfg ={},
        
        # camera sensor
        cameras_cfg={
            "eye_left": CameraCfg(
                height=720,
                width=1280,
                data_types=['rgb'],
                prim_path = "/World/CameraLeft",
                spawn=sim_utils.PinholeCameraCfg(lock_camera=False),
                offset = CameraCfg().OffsetCfg(
                    pos = (-0.3,0.033,1.6),
                    rot = (1,0,0,0),
                    convention='world')
            ),
            "eye_right": CameraCfg(
                height=720,
                width=1280,
                data_types=['rgb'],
                prim_path = "/World/CameraRight",
                spawn=sim_utils.PinholeCameraCfg(lock_camera=False),
                offset = CameraCfg().OffsetCfg(
                    pos = (-0.3,-0.033,1.6),
                    rot = (1,0,0,0),
                    convention='world')
            ),
        },
        
        # tiled camera sensor
        tiled_cameras_cfg={},

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

        random = RandomCfg(
            global_light_cfg = None,
            local_lights_cfg = None,
            rigid_objects_cfg = {
                "bottle": RigidRandomCfg(
                    random_type="range",
                    random_position=False,
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
                            PSILAB_USD_ASSET_DIR + "rigid_objects/drink-B36-V1/textures/B36.jpg",
                            PSILAB_TEXTURE_ASSET_DIR + "/20250311-092148.jpg",
                            PSILAB_TEXTURE_ASSET_DIR + "/20250311-092142.jpg",
                            PSILAB_TEXTURE_ASSET_DIR + "/20250311-092135.jpg",
                        ]
                    )
                )
            },



        )

    )



