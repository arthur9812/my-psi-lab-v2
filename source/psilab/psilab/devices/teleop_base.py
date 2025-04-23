# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-01-22
# Vesion: 1.0

import torch

from abc import abstractmethod


class TeleOpBase():
    """
    Base Class for TeleOperation Device
    """
    def __init__(self):

        self.bQuit = False
        self.bReset = False # reset flag
        self.bControl= False # human control flag
        self.bPreRecording = False  # prepareing to record flag
        self.bRecording = False # recording flag
        # teleop device ouput, default value used to reset
        self.output: dict[str,torch.Tensor] = None # type: ignore
        self.output_default: dict[str,torch.Tensor] = None # type: ignore

      
    @abstractmethod
    def reset(self):
        raise NotImplementedError
    
    @abstractmethod
    def set_default_output(self):
        raise NotImplementedError
    
    

