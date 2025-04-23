# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Isaac Lab Modules """
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors.camera import CameraCfg,Camera
from isaaclab.assets.articulation import ArticulationCfg

""" PsiLab Modules """
from psilab.assets.robot_base_cfg import RobotBaseCfg
from psilab import PSILAB_USD_ASSET_DIR


PSI_AWH_01_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # usd_path=PSILAB_USD_ASSET_DIR+"/robots/PsiRobot_AWH_01/Arm_RealMan_75_6F_NoCamera.usd",
        usd_path=PSILAB_USD_ASSET_DIR+"/robots/PsiRobot_AWH_01/Version_1/Arm_RealMan_75_6F_NoCamera.usd",
        activate_contact_sensors = False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=255,
            solver_velocity_iteration_count=32,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "joint_rev_link1": 0.0,
            "joint_rev_link2": 0.0,
            "joint_rev_link3": 0.0,
            "joint_rev_link4": 0.0,
            "joint_rev_link5": 0.0,
            "joint_rev_link6": 0.0,
            "joint_rev_link7": 0.0,
            "hand1_joint_link_1_1":0.0,
            "hand1_joint_link_1_2":0.63,
            "hand1_joint_link_1_3":0.03,
            "hand1_joint_link_2_1":3.10,
            "hand1_joint_link_2_2":1.56,
            "hand1_joint_link_3_1":3.06,
            "hand1_joint_link_3_2":1.56,
            "hand1_joint_link_4_1":3.07,
            "hand1_joint_link_4_2":1.56,
            "hand1_joint_link_5_1":3.04,
            "hand1_joint_link_5_2":1.56,
        },
    ),
    
    actuators={
        "arm1": ImplicitActuatorCfg(
            joint_names_expr=[
                "joint_rev_link1",
                "joint_rev_link2",
                "joint_rev_link3",
                "joint_rev_link4",
                "joint_rev_link5",
                "joint_rev_link6",
                "joint_rev_link7",
                ],
            # effort_limit=87.0,
            # velocity_limit=2.175,
            # stiffness=80.0,
            # damping=4.0,
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
            # ******* FengShuo Vesrion *************
            # stiffness = {
            #     "joint_rev_link1": 200.0,
            #     "joint_rev_link2": 200.0,
            #     "joint_rev_link3": 100.0,
            #     "joint_rev_link4": 100.0,
            #     "joint_rev_link5": 50.0,
            #     "joint_rev_link6": 50.0,
            #     "joint_rev_link7": 50.0,
            # },
            # damping = {
            #     "joint_rev_link1": 20.0,
            #     "joint_rev_link2": 20.0,
            #     "joint_rev_link3": 10.0,
            #     "joint_rev_link4": 10.0,
            #     "joint_rev_link5": 10.0,
            #     "joint_rev_link6": 5.0,
            #     "joint_rev_link7": 5.0,
            # },
            # effort_limit = {
            #     "joint_rev_link1": 60.0,
            #     "joint_rev_link2": 60.0,
            #     "joint_rev_link3": 30.0,
            #     "joint_rev_link4": 30.0,
            #     "joint_rev_link5": 10.0,
            #     "joint_rev_link6": 10.0,
            #     "joint_rev_link7": 10.0,
            # }
            # ******* FengShuo Vesrion *************

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
            # effort_limit=200.0,
            # velocity_limit=0.2,
            # stiffness=2e3,
            # damping=1e2,
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
    },
   
    # diff_ik_controllers = {},

    # cameras = {}

)
