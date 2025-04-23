import gymnasium as gym


##
# Register Gym environments.
##

gym.register(
    id="Psi-Replay-v1",
    entry_point=f"{__name__}.replay_env:ReplayEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.replay_env:ReplayEnvCfg",
    },
)