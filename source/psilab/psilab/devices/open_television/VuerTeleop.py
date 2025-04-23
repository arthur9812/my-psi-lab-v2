# # Copyright (c) 2022-2024, The PsiRobot Project Developers
# # Author； Feng Yun Duo
# # Date: 2025-02-06
# # Vesion: 1.0

# from .TeleVision import OpenTeleVision
# from .Preprocessor import VuerPreprocessor
# from .constants_vuer import tip_indices
# from dex_retargeting.retargeting_config import RetargetingConfig
# from pytransform3d import rotations

# from pathlib import Path
# import argparse
# import time
# import yaml
# from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore

# import numpy as np

# from ..vuer_teleop_cfg import TeleOpConfig

# class VuerTeleop:

#     def __init__(self,cfg:TeleOpConfig):
#         # eye camera resolution
#         self.resolution = (1280,1920)

#         # 图像剪裁尺寸？
#         self.crop_size_w = 0
#         self.crop_size_h = 0
#         # 图像剪裁规则？
#         self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)
#         # 图像尺寸，双眼图像沿width方向拼接，三通道rgb
#         self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
#         #
#         self.img_height, self.img_width = self.resolution_cropped[:2]
#         # 根据图像尺寸创建共享内存
#         self.shm = shared_memory.SharedMemory(create=True, size=int(np.prod(self.img_shape)* np.uint8().itemsize))
#         # image 数组
#         self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
#         # 
#         image_queue = Queue()
#         # 
#         toggle_streaming = Event()

#         # 初始化 open television,证书为相对路径，也可使用绝对路径
#         self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming,cert_file="source/psi_teleoperation/device/open_television/cert.pem", key_file="source/psi_teleoperation/device/open_television/key.pem")
#         # 初始化VuerPreprocessor
#         self.processor = VuerPreprocessor()

#         # 初始化左手 retarget 配置
#         left_hand_cfg = RetargetingConfig(
#             type="vector",
#             urdf_path="source/psi_teleoperation/device/open_television/assets/InspireHand_OY_Left/InspireHand_OY_Left.urdf",    #urdf相对路径
#             wrist_link_name="hand1_link_base",  #腕部link name
#             # 需要retarget的joint name
#             target_joint_names=[
#                 "hand1_joint_link_1_1",
#                 "hand1_joint_link_1_2",
#                 "hand1_joint_link_1_3",
#                 "hand1_joint_link_2_1",
#                 "hand1_joint_link_2_2",
#                 "hand1_joint_link_3_1",
#                 "hand1_joint_link_3_2",
#                 "hand1_joint_link_4_1",
#                 "hand1_joint_link_4_2",
#                 "hand1_joint_link_5_1",
#                 "hand1_joint_link_5_2"
#             ],  
#             # 每根手指的根部link name
#             target_origin_link_names=[
#                 "hand1_link_base",
#                 "hand1_link_base",
#                 "hand1_link_base",
#                 "hand1_link_base",
#                 "hand1_link_base",
#             ],
#             # 每根手指的远端link name
#             target_task_link_names = [
#                 "hand1_link_1_tip",
#                 "hand1_link_2_tip",
#                 "hand1_link_3_tip",
#                 "hand1_link_4_tip",
#                 "hand1_link_5_tip",
#             ],
#             scaling_factor=1.1,
#             target_link_human_indices=np.array([ [ 0, 0, 0, 0, 0 ], [ 4, 9, 14, 19, 24 ] ]),
#             low_pass_alpha = 0.5
#         )
#         # 初始化右手 retarget 配置
#         right_hand_cfg = RetargetingConfig(
#             type="vector",
#             urdf_path="source/psi_teleoperation/device/open_television/assets/InspireHand_OY_Right/InspireHand_OY_Right.urdf",  #urdf相对路径
#             wrist_link_name="hand2_link_base",  #腕部link name
#             # 需要retarget的joint name
#             target_joint_names=[
#                 "hand2_joint_link_1_1",
#                 "hand2_joint_link_1_2",
#                 "hand2_joint_link_1_3",
#                 "hand2_joint_link_2_1",
#                 "hand2_joint_link_2_2",
#                 "hand2_joint_link_3_1",
#                 "hand2_joint_link_3_2",
#                 "hand2_joint_link_4_1",
#                 "hand2_joint_link_4_2",
#                 "hand2_joint_link_5_1",
#                 "hand2_joint_link_5_2"
#             ],
#             # 每根手指的根部link name
#             target_origin_link_names=[
#                 "hand2_link_base",
#                 "hand2_link_base",
#                 "hand2_link_base",
#                 "hand2_link_base",
#                 "hand2_link_base",
#             ],
#             # 每根手指的远端link name
#             target_task_link_names = [
#                 "hand2_link_1_tip",
#                 "hand2_link_2_tip",
#                 "hand2_link_3_tip",
#                 "hand2_link_4_tip",
#                 "hand2_link_5_tip",
#             ],
#             scaling_factor=1.1,
#             target_link_human_indices=np.array([ [ 0, 0, 0, 0, 0 ], [ 4, 9, 14, 19, 24 ] ]),
#             low_pass_alpha = 0.5
#         )
        
#         # 根据配置文件创建retarget实例
#         self.left_retargeting = left_hand_cfg.build()
#         self.right_retargeting = right_hand_cfg.build()

#     def step(self):

#         head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)

#         head_rmat = head_mat[:3, :3]

#         # left_wrist_mat 手腕相对头的齐次变换矩阵
#         left_wrist_position = left_wrist_mat[:3, 3] + np.array([0.2,0.0,1.3])

#         # left_wrist_position = left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]) # -0.6, 0, 1.6 is camera position
#         left_wrist_quaternion = rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[0,1,2,3]]  #x,y,z,w

#         right_wrist_position = right_wrist_mat[:3, 3] + np.array([0.2,-0.0,1.3])
#         right_wrist_quaternion = rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[0,1,2,3]] #x,y,z,w

#         left_pose = np.concatenate([left_wrist_position,left_wrist_quaternion])
#         right_pose = np.concatenate([right_wrist_position,right_wrist_quaternion])

#         # retarget 默认关节顺序为target_joint_names，
#         # 但是usd加载到lab后joint顺序为 1_1,2_1,3_1,4_1,5_1,1_2,2_2,3_2,4_2,5_2,1_3
#         # 所以要做映射
#         left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[0,3,5,7,9,1,4,6,8,10,2]]
#         right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[0,3,5,7,9,1,4,6,8,10,2]]

#         return head_rmat, left_pose, right_pose, left_qpos, right_qpos

