# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Isaac Lab Modules """
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors.camera import CameraCfg,Camera
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

""" PsiLab Modules """
from psilab.assets.robot_base_cfg import RobotBaseCfg
from psilab import PSILAB_USD_ASSET_DIR


PSI_AWH_01_CFG = RobotBaseCfg(
    prim_path = MISSING, # type: ignore
    
    spawn=sim_utils.UsdFileCfg(
        usd_path=PSILAB_USD_ASSET_DIR+"/robots/PsiRobot_AWH_01/Version_3.0/PsiRobot_AWH_01_Left.usd",
        activate_contact_sensors = True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
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

    cameras = {
        "wrist_camera": CameraCfg(
            prim_path="/World/envs/env_[0-9]+/Robot/camera/camera",
            offset = CameraCfg.OffsetCfg(
                convention = "world"
            ),
            data_types=["rgb"],
            width=1280,
            height=720,
            spawn=None,
        )
    }

)
