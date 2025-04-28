# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Isaac Lab Modules """
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors.camera.camera_cfg import CameraCfg

""" PsiLab Modules """
from psilab import PSILAB_USD_ASSET_DIR
from psilab.assets.robot_base_cfg import RobotBaseCfg
from psilab.controllers.differential_ik_cfg import DiffIKControllerCfg


PSI_DC_01_CFG = RobotBaseCfg(

    prim_path = MISSING, # type: ignore
    
    spawn=sim_utils.UsdFileCfg(
        usd_path=PSILAB_USD_ASSET_DIR+"/robots/PsiRobot_DC_01/Version_3.0/PsiRobot_DC_01_Tuned.usd",
        activate_contact_sensors = True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
        ),
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
            
            # "arm2_joint_link1": -0.6788,
            # "arm2_joint_link2": -2.0619,
            # "arm2_joint_link3": 1.1697,
            # "arm2_joint_link4": -2.2370,
            # "arm2_joint_link5": -0.9471,
            # "arm2_joint_link6": 0.10971,
            # "arm2_joint_link7": -1.1421163,
            # "arm2_joint_link1": -0.7081,
            # "arm2_joint_link2": -2.2690,
            # "arm2_joint_link3": 1.1912,
            # "arm2_joint_link4": -1.9471,
            # "arm2_joint_link5": -0.8578,
            # "arm2_joint_link6": -0.1248,
            # "arm2_joint_link7": -1.4305,
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

    cameras = {
        "base_camera": CameraCfg(
            prim_path="/World/Robot/base_camera_rgb/base_camera_rgb",
            height=224,
            width=224,
            data_types=["rgb"],
            spawn=None
        ),
        "arm1_camera": CameraCfg(
            prim_path="/World/Robot/arm1_camera_rgb/arm1_camera_rgb",
            height=224,
            width=224,
            data_types=["rgb"],
            spawn=None
        ),
        "arm2_camera": CameraCfg(
            prim_path="/World/Robot/arm2_camera_rgb/arm2_camera_rgb",
            height=224,
            width=224,
            data_types=["rgb"],
            spawn=None
        )
    }

)
