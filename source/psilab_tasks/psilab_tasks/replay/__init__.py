# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Register Gym environments.
##
"""
Replay workflow environments.
"""

import gymnasium as gym


gym.register(
    id="Psi-Replay-v1",
    entry_point=f"{__name__}.replay_env:ReplayEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.replay_env:ReplayEnvCfg",
    },
)