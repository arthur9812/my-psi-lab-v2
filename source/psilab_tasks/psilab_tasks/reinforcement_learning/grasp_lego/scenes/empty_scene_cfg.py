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

""" Psi Lab Modules  """ 
from psilab import PSILAB_USD_ASSET_DIR
from psilab.configs.robots.psi_awh_01 import PSI_AWH_01_CFG
from psilab.scene.sence_cfg import SceneCfg
from psilab.random.random_cfg import RandomCfg,RigidRandomCfg


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
            "robot1" : PSI_AWH_01_CFG.replace(prim_path="/World/envs/env_[0-9]+/Robot"), # type: ignore
        },
        
        # static object
        static_objects_cfg = {},
        
        # rigid objects
        rigid_objects_cfg ={

            "table" : RigidObjectCfg(
                    prim_path="/World/envs/env_[0-9]+/Table", 
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=PSILAB_USD_ASSET_DIR + "/others/table_cube.usd",
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
                    usd_path=PSILAB_USD_ASSET_DIR + "/others/lego/1x2.usd",
                    scale=(1.0,1.0,1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.80, 0.64, 0.20)
                    ),
                    mass_props=MassPropertiesCfg(
                        mass = 0.01
                    ),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=64,
                        # max_linear_velocity=1.0,
                        # max_angular_velocity=180,
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
            "top_camera": TiledCameraCfg(
                prim_path="/World/envs/env_[0-9]+/top_camera",
                offset = TiledCameraCfg.OffsetCfg(
                    pos = (0.2,0.0,1.6),
                    rot = (0.707,0.0,0.707,0.0),
                    convention = "world"
                ),
                data_types=["rgb"],
                width=1280,
                height=720,
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
                width=1280,
                height=720,
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
                # "lego": sim_utils.UsdFileCfg(
                #     usd_path=PSILAB_USD_ASSET_DIR + "/others/frame_prim.usd",
                #     scale=(0.04, 0.04, 0.04),
                # ),
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
                "lego": RigidRandomCfg(
                    fake_random=False,
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

# @configclass
# class EmptySceneCfg(SceneCfg):

