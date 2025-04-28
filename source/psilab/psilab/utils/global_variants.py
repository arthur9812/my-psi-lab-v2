# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-28
# Vesion: 1.0

import time


from psilab.utils.singleton_meta import SingletonMeta

class GlobalVariant(metaclass=SingletonMeta):


    def __init__(self):
        self.is_runing = True


        
    # @property
    # def is_runing(self):
    #     return self._is_runing
    
    # def

    