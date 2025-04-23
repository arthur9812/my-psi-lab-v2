# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from collections.abc import Sequence



""" Common Modules  """ 
import torch

""" Omniverse Modules  """ 
import carb
import omni.usd
from pxr import PhysxSchema

""" IsaacSim Modules  """ 
from isaacsim.core.prims import XFormPrim

""" IsaacLab Modules  """ 
import isaaclab.sim as sim_utils
from isaaclab.scene.interactive_scene import InteractiveScene
from isaaclab.scene.interactive_scene_cfg import InteractiveSceneCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
    RigidObjectCollectionCfg,
)
from isaaclab.sensors import (
    SensorBaseCfg,
    CameraCfg,
    ContactSensorCfg, 
    FrameTransformerCfg,
    Camera
)

""" Psilab Modules  """ 
from psilab.scene.sence_cfg import SceneCfg
from psilab.assets.robot_base_cfg import RobotBaseCfg
from psilab.assets.robot_base import RobotBase
from psilab.random.random_cfg import RandomCfg,LightRandomCfg,RigidRandomCfg
from psilab.utils.random_utils import get_random_position,get_random_orientation

class Scene(InteractiveScene):
    """A scene that contains entities added to the simulation.

    The interactive scene parses the :class:`InteractiveSceneCfg` class to create the scene.
    Based on the specified number of environments, it clones the entities and groups them into different
    categories (e.g., articulations, sensors, etc.).

    Cloning can be performed in two ways:

    * For tasks where all environments contain the same assets, a more performant cloning paradigm
      can be used to allow for faster environment creation. This is specified by the ``replicate_physics`` flag.

      .. code-block:: python

          scene = InteractiveScene(cfg=InteractiveSceneCfg(replicate_physics=True))

    * For tasks that require having separate assets in the environments, ``replicate_physics`` would have to
      be set to False, which will add some costs to the overall startup time.

      .. code-block:: python

          scene = InteractiveScene(cfg=InteractiveSceneCfg(replicate_physics=False))

    Each entity is registered to scene based on its name in the configuration class. For example, if the user
    specifies a robot in the configuration class as follows:

    .. code-block:: python

        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass

        from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

        @configclass
        class MySceneCfg(InteractiveSceneCfg):

            robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    Then the robot can be accessed from the scene as follows:

    .. code-block:: python

        from isaaclab.scene import InteractiveScene

        # create 128 environments
        scene = InteractiveScene(cfg=MySceneCfg(num_envs=128))

        # access the robot from the scene
        robot = scene["robot"]
        # access the robot based on its type
        robot = scene.articulations["robot"]

    If the :class:`InteractiveSceneCfg` class does not include asset entities, the cloning process
    can still be triggered if assets were added to the stage outside of the :class:`InteractiveScene` class:

    .. code-block:: python

        scene = InteractiveScene(cfg=InteractiveSceneCfg(num_envs=128, replicate_physics=True))
        scene.clone_environments()

    .. note::
        It is important to note that the scene only performs common operations on the entities. For example,
        resetting the internal buffers, writing the buffers to the simulation and updating the buffers from the
        simulation. The scene does not perform any task specific to the entity. For example, it does not apply
        actions to the robot or compute observations from the robot. These tasks are handled by different
        modules called "managers" in the framework. Please refer to the :mod:`isaaclab.managers` sub-package
        for more details.
    """

    def __init__(self, cfg: SceneCfg):
        self.cfg = cfg

        # initiallize robot elements
        self._robots:dict[str, RobotBase] = dict()
        self._cameras:dict[str, Camera] = dict()
        self._visualizer:VisualizationMarkers = None # type: ignore
        
        super().__init__(self.cfg)
 
    def get_state(self, is_relative: bool = False) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        """Returns the state of the scene entities.

        Args:
            is_relative: If set to True, the state is considered relative to the environment origins.

        Returns:
            A dictionary of the state of the scene entities.
        """
        state = super().get_state(is_relative)

        # # robots
        # state["robot"] = dict()
        # for asset_name, robot in self._robots.items():
        #     asset_state = dict()
        #     asset_state["root_pose"] = robot.data.root_link_state_w[:, :7].clone()
        #     if is_relative:
        #         asset_state["root_pose"][:, :3] -= self.env_origins
        #     asset_state["root_velocity"] = robot.data.root_com_vel_w.clone()
        #     asset_state["joint_position"] = robot.data.joint_pos.clone()
        #     asset_state["joint_velocity"] = robot.data.joint_vel.clone()
        #     asset_state["joint_velocity"] = robot.data.joint_vel.clone()
        #     state["robot"][asset_name] = asset_state
        
        return state


    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)

        for robot in self._robots.values():
            robot.reset()

        # -- cameras
        for camera in self._cameras.values():
            camera.reset(env_ids)
        
        # random for reset
        # parse the random config
        for asset_name, random_cfg in self.cfg.random.__dict__.items():
            # skip
            if random_cfg is None:
                continue
            # 
            if isinstance(random_cfg, LightRandomCfg):
                pass
            elif isinstance(random_cfg, dict):
                # 
                for sub_asset_name, sub_random_cfg in random_cfg.items():
                    #
                    if isinstance(sub_random_cfg, RigidRandomCfg):
                        # 
                        if sub_asset_name not in self.cfg.rigid_objects_cfg.keys():
                            continue
                        #
                        if sub_random_cfg.fake_random:
                            pass
                        else:
                            # random position
                            if sub_random_cfg.random_position:
                                pos_base = self.cfg.rigid_objects_cfg[sub_asset_name].init_state.pos
                                pos_range = sub_random_cfg.position_range
                                pos = get_random_position(self.num_envs,pos_base,pos_range,self.device) + self.env_origins # type: ignore
                            else:
                                pos = torch.tensor(self.cfg.rigid_objects_cfg[sub_asset_name].init_state.pos,device = self.device).unsqueeze(0).repeat(self.num_envs,1)  + self.env_origins
                            # random orientation
                            if sub_random_cfg.random_orientation:
                                ori = get_random_orientation(self.num_envs,self.device) # type: ignore
                            else:
                                ori = torch.tensor(self.cfg.rigid_objects_cfg[sub_asset_name].init_state.rot,device=self.device).unsqueeze(0).repeat(self.num_envs,1) # type: ignore
                            pose = torch.cat((pos,ori),1)
                            self.rigid_objects[sub_asset_name].write_root_pose_to_sim(pose)
                            # random material
        pass


    def reset_to(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: Sequence[int] | None = None,
        is_relative: bool = False,
    ):
        super().reset_to(state,env_ids,is_relative)

        # """Resets the scene entities to the given state.

        # Args:
        #     state: The state to reset the scene entities to.
        #     env_ids: The indices of the environments to reset.
        #         Defaults to None (all instances).
        #     is_relative: If set to True, the state is considered relative to the environment origins.
        # """
        # if env_ids is None:
        #     env_ids = slice(None)
        # # articulations
        # for asset_name, articulation in self._articulations.items():
        #     asset_state = state["articulation"][asset_name]
        #     # root state
        #     root_pose = asset_state["root_pose"].clone()
        #     if is_relative:
        #         root_pose[:, :3] += self.env_origins[env_ids]
        #     root_velocity = asset_state["root_velocity"].clone()
        #     articulation.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
        #     articulation.write_root_com_velocity_to_sim(root_velocity, env_ids=env_ids)
        #     # joint state
        #     joint_position = asset_state["joint_position"].clone()
        #     joint_velocity = asset_state["joint_velocity"].clone()
        #     articulation.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)
        #     articulation.set_joint_position_target(joint_position, env_ids=env_ids)
        #     articulation.set_joint_velocity_target(joint_velocity, env_ids=env_ids)
        # # deformable objects
        # for asset_name, deformable_object in self._deformable_objects.items():
        #     asset_state = state["deformable_object"][asset_name]
        #     nodal_position = asset_state["nodal_position"].clone()
        #     if is_relative:
        #         nodal_position[:, :3] += self.env_origins[env_ids]
        #     nodal_velocity = asset_state["nodal_velocity"].clone()
        #     deformable_object.write_nodal_pos_to_sim(nodal_position, env_ids=env_ids)
        #     deformable_object.write_nodal_velocity_to_sim(nodal_velocity, env_ids=env_ids)
        # # rigid objects
        # for asset_name, rigid_object in self._rigid_objects.items():
        #     asset_state = state["rigid_object"][asset_name]
        #     root_pose = asset_state["root_pose"].clone()
        #     if is_relative:
        #         root_pose[:, :3] += self.env_origins[env_ids]
        #     root_velocity = asset_state["root_velocity"].clone()
        #     rigid_object.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
        #     rigid_object.write_root_com_velocity_to_sim(root_velocity, env_ids=env_ids)
        # self.write_data_to_sim()
    

    def write_data_to_sim(self):

        super().write_data_to_sim()
        
        for robot in self._robots.values():
            robot.write_data_to_sim()
        

        # print('write_data_finish:%s, write_data_robot_finish:%s' % (
        # (write_data_finish - write_data_start)*1000,
        # (write_data_robot_finish - write_data_finish)*1000,
        #     )
        # )
        
    def update(self, dt: float) -> None:
        
        super().update(dt)


        # update robot
        for robot in self._robots.values():
            robot.update(dt)

        # cameras in robot
        for camera in robot.cameras.values():
            camera.update(dt)
        
        # cameras
        for camera in self._cameras.values():
            camera.update(dt, force_recompute=not self.cfg.lazy_sensor_update)

        # print('update_finish:%s, update_finish_psi:%s' % (
        # (update_finish - update_start)*1000,
        # (update_finish_psi - update_finish)*1000,
        #     )
        # )

    def keys(self) -> list[str]:
        """Returns the keys of the scene entities.

        Returns:
            The keys of the scene entities.
        # """
        all_keys = super().keys()
        # add robot and camera
        # for asset_family in [
        #     self._robots,
        #     self._cameras,
        # ]:
        #     all_keys += list(asset_family.keys())
        return all_keys


    
    # def __getitem__(self, key: str) -> Any:
    #     """Returns the scene entity with the given key.

    #     Args:
    #         key: The key of the scene entity.

    #     Returns:
    #         The scene entity.
    #     """
    #     # check if it is a terrain
    #     if key == "terrain":
    #         return self._terrain

    #     all_keys = ["terrain"]
    #     # check if it is in other dictionaries
    #     for asset_family in [
    #         self._articulations,
    #         self._deformable_objects,
    #         self._rigid_objects,
    #         self._rigid_object_collections,
    #         self._sensors,
    #         self._extras,
    #     ]:
    #         out = asset_family.get(key)
    #         # if found, return
    #         if out is not None:
    #             return out
    #         all_keys += list(asset_family.keys())
    #     # if not found, raise error
    #     raise KeyError(f"Scene entity with key '{key}' not found. Available Entities: '{all_keys}'")

    def _add_entities_from_cfg(self):

        """
        Add scene entities from the config.
        Overwrite function in base class as config list description will cause ValueError in case class.
        """
        # ********* Super _add_entities_from_cfg
        # store paths that are in global collision filter
        self._global_prim_paths = list()
        # parse the entire scene config and resolve regex
        for asset_name, asset_cfg in self.cfg.__dict__.items():
            # skip keywords
            # note: easier than writing a list of keywords: [num_envs, env_spacing, lazy_sensor_update]
            if asset_name in InteractiveSceneCfg.__dataclass_fields__ or asset_cfg is None:
                continue
            # resolve regex
            if hasattr(asset_cfg, "prim_path"):
                asset_cfg.prim_path = asset_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
            # create asset
            if isinstance(asset_cfg, ArticulationCfg):
                self._articulations[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, DeformableObjectCfg):
                self._deformable_objects[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, RigidObjectCfg):
                self._rigid_objects[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, RigidObjectCollectionCfg):
                for rigid_object_cfg in asset_cfg.rigid_objects.values():
                    rigid_object_cfg.prim_path = rigid_object_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                self._rigid_object_collections[asset_name] = asset_cfg.class_type(asset_cfg)
                for rigid_object_cfg in asset_cfg.rigid_objects.values():
                    if hasattr(rigid_object_cfg, "collision_group") and rigid_object_cfg.collision_group == -1:
                        asset_paths = sim_utils.find_matching_prim_paths(rigid_object_cfg.prim_path)
                        self._global_prim_paths += asset_paths
            elif isinstance(asset_cfg, SensorBaseCfg):
                # Update target frame path(s)' regex name space for FrameTransformer
                if isinstance(asset_cfg, FrameTransformerCfg):
                    updated_target_frames = []
                    for target_frame in asset_cfg.target_frames:
                        target_frame.prim_path = target_frame.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                        updated_target_frames.append(target_frame)
                    asset_cfg.target_frames = updated_target_frames
                elif isinstance(asset_cfg, ContactSensorCfg):
                    updated_filter_prim_paths_expr = []
                    for filter_prim_path in asset_cfg.filter_prim_paths_expr:
                        updated_filter_prim_paths_expr.append(filter_prim_path.format(ENV_REGEX_NS=self.env_regex_ns))
                    asset_cfg.filter_prim_paths_expr = updated_filter_prim_paths_expr

                self._sensors[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, AssetBaseCfg):
                # manually spawn asset
                if asset_cfg.spawn is not None:
                    asset_cfg.spawn.func(
                        asset_cfg.prim_path,
                        asset_cfg.spawn,
                        translation=asset_cfg.init_state.pos,
                        orientation=asset_cfg.init_state.rot,
                    )
                # store xform prim view corresponding to this asset
                # all prims in the scene are Xform prims (i.e. have a transform component)
                self._extras[asset_name] = XFormPrim(asset_cfg.prim_path, reset_xform_properties=False)
            # Author: Feng Yunduo 2025-02-08 start
            # 新增配置字典类型
            elif isinstance(asset_cfg, VisualizationMarkersCfg):
                self._visualizer = VisualizationMarkers(asset_cfg)
            elif isinstance(asset_cfg, dict):
                for sub_asset_name, sub_asset_cfg in asset_cfg.items():
                    if isinstance(sub_asset_cfg, RobotBaseCfg):
                        # terrains are special entities since they define environment origins
                        self._robots[sub_asset_name] = sub_asset_cfg.class_type(sub_asset_cfg)
                        # self._articulations[sub_asset_name] = sub_asset_cfg.class_type(sub_asset_cfg)
                        # ********* Add Camera entities from the robot config ********
                        for camera_name, camera_cfg in sub_asset_cfg.cameras.items():
                            self._robots[sub_asset_name].cameras[camera_name]=Camera(camera_cfg)
                    elif isinstance(sub_asset_cfg,RigidObjectCfg):
                        self._rigid_objects[sub_asset_name] = sub_asset_cfg.class_type(sub_asset_cfg)
                    elif isinstance(sub_asset_cfg, DeformableObjectCfg):
                        self._deformable_objects[sub_asset_name] = sub_asset_cfg.class_type(sub_asset_cfg)
                    elif isinstance(sub_asset_cfg, AssetBaseCfg):
                        # manually spawn asset
                        if sub_asset_cfg.spawn is not None:
                            sub_asset_cfg.spawn.func(
                                sub_asset_cfg.prim_path,
                                sub_asset_cfg.spawn,
                                translation=sub_asset_cfg.init_state.pos,
                                orientation=sub_asset_cfg.init_state.rot,
                            )
                        # store xform prim view corresponding to this asset
                        # all prims in the scene are Xform prims (i.e. have a transform component)
                        self._extras[sub_asset_name] = XFormPrim(sub_asset_cfg.prim_path, reset_xform_properties=False)
                    elif isinstance(sub_asset_cfg, SensorBaseCfg):
                        if isinstance(sub_asset_cfg, CameraCfg):
                            self._cameras[sub_asset_name] = sub_asset_cfg.class_type(sub_asset_cfg)
                        elif isinstance(sub_asset_cfg,ContactSensorCfg):
                            self._sensors[sub_asset_name] = sub_asset_cfg.class_type(sub_asset_cfg)
            elif isinstance(asset_cfg, RandomCfg):
                # do nothing with random config
                pass
            # Author: Feng Yunduo 2025-02-08 end
            else:
                raise ValueError(f"Unknown asset config type for {asset_name}: {asset_cfg}")
            # store global collision paths
            if hasattr(asset_cfg, "collision_group") and asset_cfg.collision_group == -1:
                asset_paths = sim_utils.find_matching_prim_paths(asset_cfg.prim_path)
                self._global_prim_paths += asset_paths

        # ********* Add robot entities from the config ********
        # # parse the entire scene config and resolve regex
        # for asset_name, asset_cfg in self.cfg.__dict__.items():
        #     # skip keywords
        #     # note: easier than writing a list of keywords: [num_envs, env_spacing, lazy_sensor_update]
        #     if asset_name in SceneCfg.__dataclass_fields__ or asset_cfg is None:
        #         continue
        #     # resolve regex
        #     if hasattr(asset_cfg, "prim_path"):
        #         asset_cfg.prim_path = asset_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
        #     # # create asset
        #     if isinstance(asset_cfg, RobotCfg):
        #         # terrains are special entities since they define environment origins
        #         self._robots[asset_name] = asset_cfg.class_type(asset_cfg)
        #         # ********* Add Camera entities from the robot config ********
        #         for camera_name, camera_cfg in asset_cfg.cameras.items():
        #                 self._robots[asset_name].cameras[camera_name]=Camera(camera_cfg)
        
        # # ********* Add entities from the config list ********
        # for cfg_dict_name,cfg_dict in self.cfg.__dict__.items():
        #     if isinstance(cfg_dict,dict):
        #         for asset_name, asset_cfg in cfg_dict.items():
        #             if isinstance(asset_cfg,RigidObjectCfg):
        #                 self._rigid_objects[asset_name] = asset_cfg.class_type(asset_cfg)
        #             elif isinstance(asset_cfg, DeformableObjectCfg):
        #                 self._deformable_objects[asset_name] = asset_cfg.class_type(asset_cfg)
        #             elif isinstance(asset_cfg, AssetBaseCfg):
        #                 # manually spawn asset
        #                 if asset_cfg.spawn is not None:
        #                     asset_cfg.spawn.func(
        #                         asset_cfg.prim_path,
        #                         asset_cfg.spawn,
        #                         translation=asset_cfg.init_state.pos,
        #                         orientation=asset_cfg.init_state.rot,
        #                     )
        #                 # store xform prim view corresponding to this asset
        #                 # all prims in the scene are Xform prims (i.e. have a transform component)
        #                 self._extras[asset_name] = XFormPrimView(asset_cfg.prim_path, reset_xform_properties=False)
        #             elif isinstance(asset_cfg, SensorBaseCfg):
        #                 if isinstance(asset_cfg, CameraCfg):
        #                     self._cameras[asset_name] = asset_cfg.class_type(asset_cfg)
        #                 elif isinstance(asset_cfg,ContactSensorCfg):
        #                     self._sensors[asset_name] = asset_cfg.class_type(asset_cfg)
        #             else:
        #                 raise ValueError(f"Unknown asset config type for {asset_name}: {asset_cfg}")


    @property
    def robots(self) -> dict[str, RobotBase]:
        """A dictionary of robots in the scene."""
        return self._robots
    
    @property
    def cameras(self) -> dict[str, Camera]:
        """A dictionary of robots in the scene."""
        return self._cameras
    
    @property
    def visualizer(self) -> VisualizationMarkers:
        """A dictionary of robots in the scene."""
        return self._visualizer