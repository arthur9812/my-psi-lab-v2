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
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors.camera import CameraCfg,Camera

from isaaclab.sim.schemas.schemas_cfg import (
    MassPropertiesCfg
)

from isaaclab.assets import (
    AssetBaseCfg,
    RigidObjectCfg,
)

""" Psi Lab Modules  """ 
from psilab import PSILAB_USD_ASSET_DIR
from psilab.configs.robots.psi_awh_01 import PSI_AWH_01_CFG
from psilab.scene.sence_cfg import SceneCfg
from psilab.random.random_cfg import RandomCfg,RigidRandomCfg,MaterialRandomCfg
from psilab.assets.robot_base_cfg import RobotBaseCfg
from isaaclab.assets.articulation import ArticulationCfg

SCENE_CFG = SceneCfg(
        
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
            "robot" : RobotBaseCfg(
                prim_path = "/World/envs/env_[0-9]+/Robot",
                spawn = sim_utils.UsdFileCfg(
                    # usd_path=PSILAB_USD_ASSET_DIR+"/robots/PsiRobot_DC_01/Version_4.0/PsiRobot_DC_01_Tuned.usd",
                    usd_path=PSILAB_USD_ASSET_DIR+"/robots/PsiRobot_DC_01/Version_4.0/PsiRobot_DC_01_Tuned_Flattened_3FinAligned.usd",
                    activate_contact_sensors = True,

                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=True,
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=255,
                    )
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0),
                    rot=(1.0,0.0,0.0,0.0),
                    joint_pos={
                        "arm1_joint_link1": -0.24,
                        "arm1_joint_link2": -0.64,
                        "arm1_joint_link3": -1.52,
                        "arm1_joint_link4": -0.81,
                        "arm1_joint_link5": 0.30,
                        "arm1_joint_link6": -1.03,
                        "arm1_joint_link7": 1.35,
                        # "arm2_joint_link1": 0.24,
                        # "arm2_joint_link2": -0.64,
                        # "arm2_joint_link3": 1.52,
                        # "arm2_joint_link4": -0.81,
                        # "arm2_joint_link5": -0.30,
                        # "arm2_joint_link6": -1.03,
                        # "arm2_joint_link7": -0.36,
                        "arm2_joint_link1": 0.36939895,
                        "arm2_joint_link2": -1.42726047,
                        "arm2_joint_link3": 0.32529447,
                        "arm2_joint_link4": -0.78829542,
                        "arm2_joint_link5": -1.78686804,
                        "arm2_joint_link6": 0.85681702,
                        "arm2_joint_link7": 2.33696087,
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
                            "hand1_joint_link_2_1",
                            "hand1_joint_link_3_1",
                            "hand1_joint_link_4_1",
                            "hand1_joint_link_5_1",
                            "hand1_joint_link_1_2",
                            "hand1_joint_link_2_2",
                            "hand1_joint_link_3_2",
                            "hand1_joint_link_4_2",
                            "hand1_joint_link_5_2",
                            "hand1_joint_link_1_3"],
                        stiffness=None,
                        damping=None,

                    ),
                    "hand2": ImplicitActuatorCfg(
                        joint_names_expr=[
                            "hand2_joint_link_1_1",
                            "hand2_joint_link_2_1",
                            "hand2_joint_link_3_1",
                            "hand2_joint_link_4_1",
                            "hand2_joint_link_5_1",
                            "hand2_joint_link_1_2",
                            "hand2_joint_link_2_2",
                            "hand2_joint_link_3_2",
                            "hand2_joint_link_4_2",
                            "hand2_joint_link_5_2",
                            "hand2_joint_link_1_3"],
                        stiffness=None,
                        damping=None,

                    ),
                },
                diff_ik_controllers = {},
                eef_links={
                    "arm1":"arm1_link7",
                    "arm2":"arm2_link7"
                },
                cameras = {},
                tiled_cameras={
                   "base_camera": TiledCameraCfg(
                        prim_path="/World/envs/env_[0-9]+/Robot/base_camera_rgb/base_camera_rgb",
                        data_types=["rgb"],
                        width=640,
                        height=480,
                        spawn=None,
                    ),
                    # "arm1_camera": TiledCameraCfg(
                    #     prim_path="/World/envs/env_[0-9]+/Robot/arm1_camera_rgb/arm1_camera_rgb",
                    #     data_types=["rgb"],
                    #     width=224,
                    #     height=224,
                    #     spawn=None,
                    # ),
                    "arm2_camera": TiledCameraCfg(
                        prim_path="/World/envs/env_[0-9]+/Robot/arm2_camera_rgb/arm2_camera_rgb",
                        data_types=["rgb"],
                        width=640,
                        height=480,
                        spawn=None,
                    ),
                }
    
            )
            
        },
        
        # static object
        static_objects_cfg = {
            "ground" : AssetBaseCfg(
                prim_path="/World/envs/env_[0-9]+/Ground", 
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/envs/Grid/default_environment.usd"
                ),
                init_state = RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0), 
                    rot= (1.0,0.0, 0.0, 0.0)
                )
            )
        },
        
        # rigid objects
        rigid_objects_cfg ={

            "table" : RigidObjectCfg(
                    prim_path="/World/envs/env_[0-9]+/Table", 
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=PSILAB_USD_ASSET_DIR + "/rigid_objects/table/table_1157.usd",
                        scale=(2.0, 2.0, 1.8),
                        visual_material=None,
                        rigid_props=RigidBodyPropertiesCfg(
                            kinematic_enabled = True,
                            solver_position_iteration_count=255
                        )
                    ),
                    init_state = RigidObjectCfg.InitialStateCfg(
                        pos=(0.65, 0.0, 0.0), 
                        rot= (1.0, 0.0, 0.0, 0.0)
                    )
                ),

            "target" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target",
                spawn=sim_utils.MultiUsdFileCfg(
                    usd_path=[
                        PSILAB_USD_ASSET_DIR + "/rigid_objects/lego/1x2.usd",
                    ],
                    activate_contact_sensors = True,
                    random_choice=False,
                    scale=(1.0,1.0,1.0),
                    visual_material=None,
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=255,
                    ),
                    semantic_tags=[
                        ("class", "target")
                    ]
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.5,-0.105,0.8),
                    rot= (1,0,0,0)
                )
            ),
          
        },
        
        # rigid objects
        deformable_objects_cfg ={},
        
        # camera sensor
        cameras_cfg={},
        
        tiled_cameras_cfg = {},

        # contact sensor
        contact_sensors_cfg={
            "hand2_link_base": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand1_link_base",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Target"],
            ),
            "hand2_link_1_1": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_1_1",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Target"],
            ),
            "hand2_link_1_2": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_1_2",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Target"],
            ),
            "hand2_link_1_3": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_1_3",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Target"],
            ),
            "hand2_link_2_1": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_2_1",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Target"],
            ),
            "hand2_link_2_2": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_2_2",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Target"],
            ),
            "hand2_link_3_1": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_3_1",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Target"],
            ),
            "hand2_link_3_2": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_3_2",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Target"],
            ),
            "hand2_link_4_1": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_4_1",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Target"],
            ),
            "hand2_link_4_2": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_4_2",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Target"],
            ),
            "hand2_link_5_1": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_5_1",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Target"],
            ),
            "hand2_link_5_2": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/hand2_link_5_2",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_[0-9]+/Target"],
            ),
        },

        # debug marker
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Markers",
            markers={
                "target_position": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/markers/frame_prim.usd",
                    scale=(0.05, 0.05, 0.05),
                ),
                "thumb": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/markers/frame_prim.usd",
                    scale=(0.01, 0.01, 0.01),
                ),
                "index": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/markers/frame_prim.usd",
                    scale=(0.01, 0.01, 0.01),
                ),
                "middle": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/markers/frame_prim.usd",
                    scale=(0.01, 0.01, 0.01),
                ),
                "ring": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/markers/frame_prim.usd",
                    scale=(0.01, 0.01, 0.01),
                ),
                "pinky": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/markers/frame_prim.usd",
                    scale=(0.01, 0.01, 0.01),
                ),
                "lego": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/markers/frame_prim.usd",
                    scale=(0.04, 0.04, 0.04),
                ),
                # "middle_point": sim_utils.UsdFileCfg(
                #     usd_path=PSILAB_USD_ASSET_DIR + "/markers/frame_prim.usd",
                #     scale=(0.01, 0.01, 0.01),
                # ),


            },

        ),

        random = RandomCfg(
            global_light_cfg = None,
            local_lights_cfg = None,
            rigid_objects_cfg = {
                "target": RigidRandomCfg(
                    random_type="range",
                    random_position=True,
                    random_orientation=True,
                    random_material=True,
                    position_range=[0.12,0.155,0.0],
                    position_list=None,
                    orientation_list=None,
                    material_cfg=MaterialRandomCfg(
                        enable_random=True,
                        shader_path="/World/envs/env_[0-9]+/Target/Looks/material/shader",
                        random_type="range",
                        material_type="color",
                        color_range=[
                            [0,0,0],
                            [255,255,255] # type: ignore
                        ],
                        color_list=[], # type: ignore,
                        texture_list=[]


                    )
                )
            },

        )

    )
