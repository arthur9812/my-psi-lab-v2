import gymnasium as gym
from . import scenes

##
# Register Gym environments.
##

gym.register(
    id="Psi-IL-Grasp-Rigid-v1",
    entry_point=f"{__name__}.grasp_rigid_env:GraspRigidEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_rigid_env:GraspRigidEnvCfg",
        "scene_cfg_entry_point":f"{scenes.__name__}.room_cfg:SCENE_CFG",
        # "scene_cfg_entry_point":f"{scenes.__name__}:room_scene_cfg.json",
    },
)