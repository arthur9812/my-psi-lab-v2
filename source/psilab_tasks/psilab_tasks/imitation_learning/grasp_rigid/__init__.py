import gymnasium as gym


##
# Register Gym environments.
##

gym.register(
    id="Psi-IL-Grasp-Rigid-v1",
    entry_point=f"{__name__}.grasp_rigid_env:GraspRigidEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_rigid_env:GraspRigidEnvCfg",
    },
)