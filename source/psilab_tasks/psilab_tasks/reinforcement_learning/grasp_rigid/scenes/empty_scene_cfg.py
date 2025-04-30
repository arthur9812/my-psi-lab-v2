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

from isaaclab.sim.schemas.schemas_cfg import (
    MassPropertiesCfg
)

from isaaclab.assets import (
    AssetBaseCfg,
    RigidObjectCfg,
)
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg


""" Psi Lab Modules  """ 
from psilab import PSILAB_USD_ASSET_DIR
from psilab.configs.robots.psi_dc_01 import PSI_DC_01_CFG
from psilab.scene.sence_cfg import SceneCfg
from psilab.random.random_cfg import RandomCfg,RigidRandomCfg


EMPTY_SCENE_CFG = SceneCfg(
        
        num_envs = 1, 
        env_spacing=10.0, 
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
                prim_path="/World/envs/env_[0-9]+/Robot",
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0),
                    rot=(1.0,0.0,0.0,0.0),
                    joint_pos={
                        # "arm1_joint_link1": -0.24,
                        # "arm1_joint_link2": -0.64,
                        # "arm1_joint_link3": -1.52,
                        # "arm1_joint_link4": -0.81,
                        # "arm1_joint_link5": 0.30,
                        # "arm1_joint_link6": -1.03,
                        # "arm1_joint_link7": 1.35,
                        "arm2_joint_link1": 0.41,
                        "arm2_joint_link2": -1.57,
                        "arm2_joint_link3": 0.50,
                        "arm2_joint_link4": -1.80,
                        "arm2_joint_link5": -0.95,
                        "arm2_joint_link6": 0.92,
                        "arm2_joint_link7": -0.35,
                        # "hand1_joint_link_1_1":0.0,
                        # "hand1_joint_link_1_2":0.63,
                        # "hand1_joint_link_1_3":0.03,
                        # "hand1_joint_link_2_1":3.10,
                        # "hand1_joint_link_2_2":1.56,
                        # "hand1_joint_link_3_1":3.06,
                        # "hand1_joint_link_3_2":1.56,
                        # "hand1_joint_link_4_1":3.06,
                        # "hand1_joint_link_4_2":1.56,
                        # "hand1_joint_link_5_1":3.04,
                        # "hand1_joint_link_5_2":1.56,
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
                diff_ik_controllers ={}
                ),
                
        },
        
        # static object
        static_objects_cfg = {
            "ground" : AssetBaseCfg(
                prim_path="/World/envs/env_[0-9]+/Ground", 
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others//Grid/default_environment.usd"
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
                        usd_path=PSILAB_USD_ASSET_DIR + "/others/table_1157.usd",
                        scale=(2.0, 2.0, 1.8),
                        visual_material=None,
                        rigid_props=RigidBodyPropertiesCfg(
                            kinematic_enabled = True,
                        )
                    ),
                    init_state = RigidObjectCfg.InitialStateCfg(
                        pos=(0.65, 0.0, 0.0), 
                        rot= (1.0, 0.0, 0.0, 0.0)
                    )
                ),

            # "lego" : RigidObjectCfg(
            #     prim_path="/World/envs/env_[0-9]+/Lego",
            #     spawn=sim_utils.UsdFileCfg(
            #         usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2.usd",
            #         scale=(1.0,1.0,1.0),
            #         visual_material=sim_utils.PreviewSurfaceCfg(
            #             diffuse_color=(0.80, 0.64, 0.20)
            #         ),
            #         mass_props=MassPropertiesCfg(
            #             mass = 0.01
            #         ),
            #         rigid_props=RigidBodyPropertiesCfg(
            #             solver_position_iteration_count=255,
            #         ),
            
            #     ),
            #     init_state=RigidObjectCfg.InitialStateCfg(
            #         pos=(0.5,-0.15,0.85),
            #         rot= (1,0,0,0)
            #     )
            # ),
       
            "target1" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target1",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_1.usd",
                    scale=(1.0,1.0,1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.64, 0.20)
                    ),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=255
                        )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.7,-0.15,0.85),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),
                        
            "target2" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target2",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_2.usd",
                    scale=(1.0,1.0,1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.64, 0.20)
                    ),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=255
                        )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.6,-0.1,0.85),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),
            
            "target3" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target3",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_3.usd",
                    scale=(1.0,1.0,1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.64, 0.20)
                    ),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=255
                        )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.7,-0.05,0.95),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),
            
            "target4" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target4",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_4.usd",
                    scale=(1.0,1.0,1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.64, 0.20)
                    ),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=255
                        )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.6,-0.0,1.0),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),
            
            "target5" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target5",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_5.usd",
                    scale=(1.0,1.0,1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.64, 0.20)
                    ),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=255
                        )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.6,0.05,0.85),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),
            "target6" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target6",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_6.usd",
                    scale=(1.0,1.0,1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.64, 0.20)
                    ),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=255
                        )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.6,0.1,0.85),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),
            "target7" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target7",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_7.usd",
                    scale=(1.0,1.0,1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.64, 0.20)
                    ),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=255
                        )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.6,0.15,0.85),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),
            "target8" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target8",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_8.usd",
                    scale=(1.0,1.0,1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.64, 0.20)
                    ),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=255
                        )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.65,-0.15,0.85),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),
            "target9" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target9",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_9.usd",
                    scale=(1.0,1.0,1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.64, 0.20)
                    ),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=255
                        )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.75,-0.15,0.85),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),
            "target10" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target10",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_10.usd",
                    scale=(1.0,1.0,1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.64, 0.20)
                    ),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=255
                        )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.8,-0.15,0.85),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),
            "target11" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target11",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_11.usd",
                    scale=(1.0,1.0,1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.64, 0.20)
                    ),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=255
                        )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.8,-0.1,0.85),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),
            "target12" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target12",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_12.usd",
                    scale=(1.0,1.0,1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.64, 0.20)
                    ),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=255
                        )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.85,-0.1,0.85),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),
            "target13" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target13",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_13.usd",
                    scale=(1.0,1.0,1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.64, 0.20)
                    ),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=255
                        )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.85,-0.05,0.85),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),
            "target14" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target14",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_14.usd",
                    scale=(1.0,1.0,1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.64, 0.20)
                    ),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=255
                        )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.85,-0.0,0.85),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),
            "target15" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target15",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_15.usd",
                    scale=(1.0,1.0,1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.64, 0.20)
                    ),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=255
                        )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.6,-0.15,0.85),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),
            "target16" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target16",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_16.usd",
                    scale=(1.0,1.0,1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.64, 0.20)
                    ),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=255
                        )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.85,0.05,0.85),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),
            "target17" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target17",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_17.usd",
                    scale=(1.0,1.0,1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.64, 0.20)
                    ),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=255
                        )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.9,0.15,0.85),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),
            "target18" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target18",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_18.usd",
                    scale=(1.0,1.0,1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.64, 0.20)
                    ),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=255
                        )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.9,0.05,0.85),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),
            "target19" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target19",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_19.usd",
                    scale=(1.0,1.0,1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.64, 0.20)
                    ),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=255
                        )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.9,-0.15,0.85),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),
            "target20" : RigidObjectCfg(
                prim_path="/World/envs/env_[0-9]+/Target20",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_20.usd",
                    scale=(1.0,1.0,1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.64, 0.20)
                    ),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=255
                        )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.9,-0.05,0.85),
                    rot= (1.0,0.0,0.0,0.0)
                )
            ),
            # "target21" : RigidObjectCfg(
            #     prim_path="/World/envs/env_[0-9]+/Lego21",
            #     spawn=sim_utils.UsdFileCfg(
            #         usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_.usd",
            #         scale=(1.0,1.0,1.0),
            #         visual_material=sim_utils.PreviewSurfaceCfg(
            #             diffuse_color=(0.80, 0.64, 0.20)
            #         ),
            #         mass_props=MassPropertiesCfg(
            #             mass = 0.01
            #         ),
            #         rigid_props=RigidBodyPropertiesCfg(
            #                 solver_position_iteration_count=255
            #             )
            #     ),
            #     init_state=RigidObjectCfg.InitialStateCfg(
            #         pos=(0.9,-0.05,0.85),
            #         rot= (1.0,0.0,0.0,0.0)
            #     )
            # ),
            # "target22" : RigidObjectCfg(
            #     prim_path="/World/envs/env_[0-9]+/Lego22",
            #     spawn=sim_utils.UsdFileCfg(
            #         usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_.usd",
            #         scale=(1.0,1.0,1.0),
            #         visual_material=sim_utils.PreviewSurfaceCfg(
            #             diffuse_color=(0.80, 0.64, 0.20)
            #         ),
            #         mass_props=MassPropertiesCfg(
            #             mass = 0.01
            #         ),
            #         rigid_props=RigidBodyPropertiesCfg(
            #                 solver_position_iteration_count=255
            #             )
            #     ),
            #     init_state=RigidObjectCfg.InitialStateCfg(
            #         pos=(0.9,-0.05,0.85),
            #         rot= (1.0,0.0,0.0,0.0)
            #     )
            # ),
            # "target23" : RigidObjectCfg(
            #     prim_path="/World/envs/env_[0-9]+/Lego23",
            #     spawn=sim_utils.UsdFileCfg(
            #         usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_.usd",
            #         scale=(1.0,1.0,1.0),
            #         visual_material=sim_utils.PreviewSurfaceCfg(
            #             diffuse_color=(0.80, 0.64, 0.20)
            #         ),
            #         mass_props=MassPropertiesCfg(
            #             mass = 0.01
            #         ),
            #         rigid_props=RigidBodyPropertiesCfg(
            #                 solver_position_iteration_count=255
            #             )
            #     ),
            #     init_state=RigidObjectCfg.InitialStateCfg(
            #         pos=(0.9,-0.05,0.85),
            #         rot= (1.0,0.0,0.0,0.0)
            #     )
            # ),
            # "target24" : RigidObjectCfg(
            #     prim_path="/World/envs/env_[0-9]+/Lego24",
            #     spawn=sim_utils.UsdFileCfg(
            #         usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_.usd",
            #         scale=(1.0,1.0,1.0),
            #         visual_material=sim_utils.PreviewSurfaceCfg(
            #             diffuse_color=(0.80, 0.64, 0.20)
            #         ),
            #         mass_props=MassPropertiesCfg(
            #             mass = 0.01
            #         ),
            #         rigid_props=RigidBodyPropertiesCfg(
            #                 solver_position_iteration_count=255
            #             )
            #     ),
            #     init_state=RigidObjectCfg.InitialStateCfg(
            #         pos=(0.9,-0.05,0.85),
            #         rot= (1.0,0.0,0.0,0.0)
            #     )
            # ),
            # "target25" : RigidObjectCfg(
            #     prim_path="/World/envs/env_[0-9]+/Lego25",
            #     spawn=sim_utils.UsdFileCfg(
            #         usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2_.usd",
            #         scale=(1.0,1.0,1.0),
            #         visual_material=sim_utils.PreviewSurfaceCfg(
            #             diffuse_color=(0.80, 0.64, 0.20)
            #         ),
            #         mass_props=MassPropertiesCfg(
            #             mass = 0.01
            #         ),
            #         rigid_props=RigidBodyPropertiesCfg(
            #                 solver_position_iteration_count=255
            #             )
            #     ),
            #     init_state=RigidObjectCfg.InitialStateCfg(
            #         pos=(0.9,-0.05,0.85),
            #         rot= (1.0,0.0,0.0,0.0)
            #     )
            # ),
         
        },
        
        # rigid objects
        deformable_objects_cfg ={},
        
        # camera sensor
        cameras_cfg={
            # "top_camera": TiledCameraCfg(
            #     prim_path="/World/envs/env_[0-9]+/top_camera",
            #     offset = TiledCameraCfg.OffsetCfg(
            #         pos = (0.2,0.0,1.6),
            #         rot = (0.707,0.0,0.707,0.0),
            #         convention = "world"
            #     ),
            #     data_types=["rgb"],
            #     width=1280,
            #     height=720,
            #     spawn=PinholeCameraCfg(),
            # ),
            # "front_camera": TiledCameraCfg(
            #     prim_path="/World/envs/env_[0-9]+/front_camera",
            #     offset = TiledCameraCfg.OffsetCfg(
            #         pos = (1.0,0.0,0.7),
            #         rot = (0.0,0.0,0.0,1.0),
            #         convention = "world"
            #     ),
            #     data_types=["rgb"],
            #     width=1280,
            #     height=720,
            #     spawn=PinholeCameraCfg(),
            # ),
        },
        
        # contact sensor
        contact_sensors_cfg={
            "right_hand": ContactSensorCfg(
                prim_path="/World/envs/env_[0-9]+/Robot/InspireHand_OY_Right/hand2_link_.*",
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=[],
            ),
        },

        # debug marker
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Markers",
            markers={
                "thumb": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/frame_prim.usd",
                    scale=(0.01, 0.01, 0.01),
                ),
                "index": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/frame_prim.usd",
                    scale=(0.01, 0.01, 0.01),
                ),
                "middle": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/frame_prim.usd",
                    scale=(0.01, 0.01, 0.01),
                ),
                "ring": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/frame_prim.usd",
                    scale=(0.01, 0.01, 0.01),
                ),
                "pinky": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/frame_prim.usd",
                    scale=(0.01, 0.01, 0.01),
                ),
                "lego": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/frame_prim.usd",
                    scale=(0.04, 0.04, 0.04),
                ),
                "middle_point": sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/frame_prim.usd",
                    scale=(0.01, 0.01, 0.01),
                ),


            },

        ),

        random = RandomCfg(
            global_light_cfg = None,
            local_lights_cfg = None,
            rigid_objects_cfg = {
                # "lego": RigidRandomCfg(
                #     random_type="range",
                #     random_position=True,
                #     random_orientation=True,
                #     random_material=False,
                #     position_range=[0.05,0.05,0.0],
                #     position_list=None,
                #     orientation_list=None
                # ),
                # "bottle": RigidRandomCfg(
                #     random_type="range",
                #     random_position=True,
                #     random_orientation=True,
                #     random_material=False,
                #     position_range=[0.05,0.05,0.0],
                #     position_list=None,
                #     orientation_list=None
                # )
            },

        )

    )
