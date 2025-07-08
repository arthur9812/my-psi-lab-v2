# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

import wandb


from psilab.utils.singleton_meta import SingletonMeta

class WandbLog(metaclass=SingletonMeta):


    def __init__(self):
        self.log_data : dict[ str, float]= {} # type: ignore 
        self.step : int = 0
        self._init = False


    def init_wandb(self, project:str, name:str, tags:list[str]=[]):
        if len(tags) > 0:
            wandb.init(project=project, name=name, tags=tags)  
        else:
            wandb.init(project=project, name=name)  
        self.project = project
        self.name = name
        self._init = True
        self.artifact = {}
        
    def init_artifact(self, artifact_name:str, artifact_type:str):
        self.artifact[artifact_type] = wandb.Artifact(artifact_name, type=artifact_type)

    def upload_artifacts_from_path(self, artifact_type:str, path:str):
        self.artifact[artifact_type].add_dir(path)
        wandb.log_artifact(self.artifact[artifact_type])
        
    def set_data(self, key:str, value:float):
        self.log_data[key] = value

    def get_data(self, key:str)->float:
        return self.log_data[key]
    
    def get_step(self)->int:
        return self.step
    
    def upload(self,key:str):

        if key not in self.log_data.keys() or not self._init:
            return
        # if len(self.log_data.keys())==0:
        #     return
        self.step += 1
        wandb.log(
            { key:self.log_data[key]}, 
            step = self.step
            )
        
    def upload_all(self):
        if not self._init:
            return
        self.step += 1
        wandb.log(self.log_data, step = self.step)
    