# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from __future__ import annotations
from typing import Any

""" Common Modules  """ 
import torch
from datetime import datetime

""" IsaacLab Modules  """ 
from isaaclab.envs.common import VecEnvStepReturn

""" PsiLab Modules  """ 
from psilab.devices.vuer_tp import VuerTp
from psilab.envs.tp_env_cfg import TPEnvCfg
from psilab.envs.rl_env import RLEnv
from psilab.utils.voice_utils import ESpeak
from psilab.utils.data_collect_utils import create_empty_data

class TPEnv(RLEnv):
    """The tele operation environment class."""

    def __init__(self, cfg: TPEnvCfg, render_mode: str | None = None, **kwargs):



        # 实例化语音类
        self.voice = ESpeak(speed=300, voice='zh')

        # 实例化vuer    
        self.vuer = VuerTp(cfg.device_cfg) # type: ignore

        # 眼部相机位置需要根据robot和teleop config计算得
        camere_eye_left_pos=(
            cfg.scene.robots_cfg["robot1"].init_state.pos[0] + cfg.device_cfg.head_pos[0] + cfg.device_cfg.eye_left_offset[0], # type: ignore
            cfg.scene.robots_cfg["robot1"].init_state.pos[1] + cfg.device_cfg.head_pos[1] + cfg.device_cfg.eye_left_offset[1], # type: ignore
            cfg.scene.robots_cfg["robot1"].init_state.pos[2] + cfg.device_cfg.head_pos[2] + cfg.device_cfg.eye_left_offset[2], # type: ignore                                
        ),

        camere_eye_right_pos=(
            cfg.scene.robots_cfg["robot1"].init_state.pos[0] + cfg.device_cfg.head_pos[0] + cfg.device_cfg.eye_right_offset[0], # type: ignore
            cfg.scene.robots_cfg["robot1"].init_state.pos[1] + cfg.device_cfg.head_pos[1] + cfg.device_cfg.eye_right_offset[1], # type: ignore
            cfg.scene.robots_cfg["robot1"].init_state.pos[2] + cfg.device_cfg.head_pos[2] + cfg.device_cfg.eye_right_offset[2], # type: ignore                                   
        ),

        cfg.scene.cameras_cfg["eye_left"].offset.pos = camere_eye_left_pos # type: ignore
        cfg.scene.cameras_cfg["eye_right"].offset.pos = camere_eye_right_pos # type: ignore
        #
        super().__init__(cfg, render_mode, **kwargs)
        #
        self.cfg = cfg
        # 
        self._data = create_empty_data(self,self.cfg)
        
        # change output folder with date and time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cfg.ouput_folder+=f"{timestamp}/"

        # fake state
        self._obs_zero = {
            "policy":torch.zeros((self.num_envs,self.cfg.observation_space),device=self.device), # type: ignore
            "critic":torch.zeros((self.num_envs,self.cfg.observation_space),device=self.device) # type: ignore
        }
        self._reward_zero = torch.zeros(self.num_envs,device=self.device) # type: ignore
        self._reset_zero = torch.tensor([0 for i in range(self.num_envs)], device=self.device)
        self._dones_zero = torch.tensor([0 for i in range(self.num_envs)], device=self.device)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # return observations, rewards, resets and extras
        return self._obs_zero, self._reward_zero, self._reset_zero, self.reset_time_outs, dict()

    def sim_step(self):
        # robot step to compute ik and ..., to set joint target
        for robot in self.scene.robots.values():
            robot.step()
        
        # set actions into simulator
        self.scene.write_data_to_sim()

        # simulate
        self.sim.step(render=True)
        # render between steps only if the GUI or an RTX sensor needs it
        # note: we assume the render interval to be the shortest accepted rendering interval.
        #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
        # if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
        #     self.sim.render()
        # update buffers at sim dt
        self.scene.update(dt=self.physics_dt)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        # reset scene
        self.scene.reset()
        # 
        self.vuer.reset()
        # clear data
        self._data = create_empty_data(self,self.cfg)
        # clear cuda cache
        torch.cuda.empty_cache()
        # 
        return super().reset()



    """
    Functions for RL which is useless in Tele Operarion Env
    """
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        pass

    def _get_observations(self) -> dict:
        return self._obs_zero

    def _get_rewards(self) -> torch.Tensor:
       return self._reward_zero

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._reset_zero,self._dones_zero

