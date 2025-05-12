import gymnasium as gym

from . import agents
from . import scenes

##
# Register Gym environments.
##

gym.register(
    id="Psi-RL-Grasp-Rigid-v1",
    entry_point=f"{__name__}.grasp_rigid_env:GraspRigidEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_rigid_env:GraspRigidEnvCfg",
        # "scene_cfg_entry_point":f"{scenes.__name__}:empty_scene_cfg.json",
        "scene_cfg_entry_point":f"{scenes.__name__}.empty_scene_cfg:EMPTY_SCENE_CFG",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml"
    },
)