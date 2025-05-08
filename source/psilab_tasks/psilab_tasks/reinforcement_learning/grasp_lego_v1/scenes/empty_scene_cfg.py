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
from psilab.random.random_cfg import RandomCfg,RigidRandomCfg
from psilab.assets.robot_base_cfg import RobotBaseCfg
from isaaclab.assets.articulation import ArticulationCfg

EMPTY_SCENE_CFG = SceneCfg(
        
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
                    usd_path=PSILAB_USD_ASSET_DIR+"/robots/PsiRobot_AWH_01/Version_3.0/PsiRobot_AWH_01_Left.usd",
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=False,
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=255,
                    )
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(-0.3, 0.0, 0.6),
                    rot=(1.0,0.0,0.0,0.0),
                    joint_pos={
                        "joint_rev_link1": 0.0,
                        "joint_rev_link2": 0.45,
                        "joint_rev_link3": 0.0,
                        "joint_rev_link4": 1.78,
                        "joint_rev_link5": 0.0,
                        "joint_rev_link6": -0.5,
                        "joint_rev_link7": -2.54,
                        "hand1_joint_link_1_1":0.0,
                        "hand1_joint_link_2_1":3.10,
                        "hand1_joint_link_3_1":3.06,
                        "hand1_joint_link_4_1":3.07,
                        "hand1_joint_link_5_1":3.04,
                        "hand1_joint_link_1_2":0.63,
                        "hand1_joint_link_2_2":1.56,
                        "hand1_joint_link_3_2":1.56,
                        "hand1_joint_link_4_2":1.56,
                        "hand1_joint_link_5_2":1.56,
                        "hand1_joint_link_1_3":0.03,
                    }
                ),
                                    
                actuators={
                    "arm": ImplicitActuatorCfg(
                        joint_names_expr=[
                            "joint_rev_link1",
                            "joint_rev_link2",
                            "joint_rev_link3",
                            "joint_rev_link4",
                            "joint_rev_link5",
                            "joint_rev_link6",
                            "joint_rev_link7",
                        ],
                        stiffness=None,
                        damping=None,
                    ),
                    "hand": ImplicitActuatorCfg(
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
                            "hand1_joint_link_1_3",
                        ],
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
                    "wrist_camera": TiledCameraCfg(
                        prim_path="/World/envs/env_[0-9]+/Robot/camera/camera",
                        offset = TiledCameraCfg.OffsetCfg(
                            convention = "world"
                        ),
                        data_types=["rgb"],
                        width=640,
                        height=480,
                        spawn=None,
                    )
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
                        usd_path=PSILAB_USD_ASSET_DIR + "/rigid_objects/table/table_cube.usd",
                        scale=(1.5, 1.0, 0.6),
                        visual_material=None,
                        rigid_props=RigidBodyPropertiesCfg(
                            kinematic_enabled = True,
                        )
                    ),
                    init_state = RigidObjectCfg.InitialStateCfg(
                        pos=(0.0, 0.0, 0.0), 
                        rot= (1.0, 0.0, 0.0, 0.0)
                    )
                ),

            "lego" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Lego",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/rigid_objects/lego/1x2.usd",
                    scale=(1.0,1.0,1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.64, 0.20)
                    ),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=255,
                    ),
            
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.2,0,0.7),
                    rot= (1,0,0,0)
                )
            ),
          
        },
        
        # rigid objects
        deformable_objects_cfg ={},
        
        # camera sensor
        cameras_cfg={
            
        },
        
        tiled_cameras_cfg = {
            "top_camera": TiledCameraCfg(
                prim_path="/World/envs/env_[0-9]+/top_camera",
                offset = TiledCameraCfg.OffsetCfg(
                    pos = (0.2,0.0,1.6),
                    rot = (0.707,0.0,0.707,0.0),
                    convention = "world"
                ),
                data_types=["rgb"],
                width=640,
                height=480,
                spawn=PinholeCameraCfg(),
            ),
            "front_camera": TiledCameraCfg(
                prim_path="/World/envs/env_[0-9]+/front_camera",
                offset = TiledCameraCfg.OffsetCfg(
                    pos = (1.0,0.0,0.7),
                    rot = (0.0,0.0,0.0,1.0),
                    convention = "world"
                ),
                data_types=["rgb"],
                width=640,
                height=480,
                spawn=PinholeCameraCfg(),
            ),
        },

        # contact sensor
        contact_sensors_cfg={},

        # debug marker
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Markers",
            markers={
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
                "middle_point": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/markers/frame_prim.usd",
                    scale=(0.01, 0.01, 0.01),
                ),


            },

        ),

        random = RandomCfg(
            global_light_cfg = None,
            local_lights_cfg = None,
            rigid_objects_cfg = {
                "lego": RigidRandomCfg(
                    random_type="range",
                    random_position=True,
                    random_orientation=True,
                    random_material=False,
                    position_range=[0.1,0.1,0.0],
                    position_list=None,
                    orientation_list=None
                )
            },

        )

    )
