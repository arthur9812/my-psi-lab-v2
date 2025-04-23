# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

import os
import sys
import yaml  # 用于读取配置文件
import dill  # 用于加载模型
import hydra  # 用于加载diffusion policy
import torch
import ssl
import requests
import torch.nn.functional as F

# Add Diffusion Policy Project Path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append("/home/admin01/Work/00-DiffusionPolicy/diffusion_policy") 

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from psi_dp.workspace.train_diffusion_transformer_timm_workspace import TrainDiffusionTransformerTimmWorkspace

#1.load policy （base policy and res policy）
def load_diffusion_policy_model(checkpoint_path):

    ssl._create_default_https_context = ssl._create_unverified_context

    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

    # 修改 huggingface_hub 的设置
    # import osct
    os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
    ckpt_path = checkpoint_path
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    # 打印目标类路径以进行调试
    print(f"Target class path: {cfg._target_}")

    # 如果需要，手动修改目标类路径
    # cfg._target_ = "../../../DexDiffusionPolicy/diffusion_policy/workspace/train_diffusion_transformer_timm_workspace.TrainDiffusionTransformerTimmWorkspace"

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        # if cfg.training.use_ema:
        #     policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 16 # DDIM inference iterations
        policy.n_action_steps = 8
    return policy

def process_image(img:torch.Tensor):
    # # 转换为torch tensor并保持float类型
    # print(isinstance(img, torch.Tensor))
    # if isinstance(img, torch.Tensor):
    #     img = img[..., :3].float()  # 只保留RGB通道
    # else:
    #     img = torch.from_numpy(img[..., :3]).float()  # 转换为tensor并保留RGB通道
    
    # 只保留RGB通道
    img = img[...,:3].float()  
    # 调整通道顺序 [H, W, C] -> [C, H, W]
    img = img.permute(2, 0, 1)  
    # img = img.permute(2, 1, 0)  # 调整通道顺序

    # 归一化到[0,1]并调整尺寸
    img = F.interpolate(
        img.unsqueeze(0) / 255.0,  # 添加batch维度并归一化
        size=(224, 224),           # 调整到模型期望的尺寸
        mode='bilinear',
        align_corners=False
    )
    # # 归一化到[0,1]并调整尺寸
    # img = F.interpolate(
    #     img.unsqueeze(0) / 255.0,  # 添加batch维度并归一化
    #     size=(224, 224),           # 调整到模型期望的尺寸
    #     mode='bilinear',
    #     align_corners=False
    # ).squeeze(0)  # 移除batch维度
    # print(f"Final processed image shape: {img.shape}")
    return img

