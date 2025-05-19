import gymnasium as gym

from . import agents
from . import scenes

##
# Register Gym environments.
##

# Psi-RL-Grasp-Lego-v1 use Psi DC 01 as robot, lego is 1x2 
gym.register(
    id="Psi-RL-Grasp-Lego-v2",
    entry_point=f"{__name__}.grasp_lego_env:GraspLegoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_lego_env:GraspLegoEnvCfg",
        "scene_cfg_entry_point":f"{scenes.__name__}.empty_cfg:SCENE_CFG",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml"
    },
)