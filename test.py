import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.linalg import svd

@torch.jit.script
def planar_force_closure(
    p: torch.Tensor,            # (B,3,3)  指尖位置  (x,y,z)
    n: torch.Tensor,            # (B,3,3)  指尖内法向 (单位向量)
    eps0: float = 0.05          # 缩放参数
) -> torch.Tensor:              # (B,)      批量奖励
    """
    6d扳手力闭包的简化模型[Fx,Fy,Fz,τx,τy,τz]->[Fx,Fy,τz]
    只考虑xy平面的扰动
    """
    # ---- 1. 取 x、y 分量 ----
    n_xy = n[:, :, :2]                 # (B,3,2)  提取法向的 x、y
    px, py = p[:, :, 0], p[:, :, 1]    # (B,3)    指尖 x、y 坐标
    wx, wy = n_xy[:, :, 0], n_xy[:, :, 1]  # (B,3) 法向的 x、y 分量

    # ---- 2. 计算 z 方向力矩 τ_z = (p × n)_z ----
    tau_z = px * wy - py * wx          # (B,3)    行列式公式

    # ---- 3. 拼成平面扳手列 w_planar ----
    w_planar = torch.stack([wx, wy, tau_z], dim=-1)  # (B,3,3)

    # ---- 4. 列向量单位化 ----
    w_planar = torch.nn.functional.normalize(w_planar, dim=-1)

    # ---- 5. 变成 3×3 抓取矩阵 G ----
    G = w_planar.permute(0, 2, 1)      # (B,3,3)  shape: (B, 行=3, 列=3)

    # ---- 6. 奇异值分解 SVD ----
    _, S, _ = svd(G)                   # S: (B,3)  降序奇异值

    # ---- 7. 提取最小奇异值 σ_min ----
    sigma_min = S[:, -1]               # (B,)     每批次最小值

    # ---- 8. 平滑映射成奖励 ----
    return torch.tanh(sigma_min / eps0)

# -----------------------------------------------------------
# 1. 四元数 → 旋转矩阵  (w, x, y, z)  →  R ∈ SO(3)
# -----------------------------------------------------------
@torch.jit.script
def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    将四元数 (w,x,y,z) 转为旋转矩阵。
    参数
    ----
    q : Tensor[..., 4]

    返回
    ----
    R : Tensor[..., 3, 3]
    """
    eps: float = 1e-8
    # --- 归一化 ---
    norm = torch.norm(q, p=2, dim=-1, keepdim=True)
    q = q / torch.clamp(norm, min=eps)

    # 拆分分量（TorchScript 不支持一次性“多变量解包”）
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    # --- 预计算重复项 ---
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    two: float = 2.0  # 方便和论文公式对照

    # 每个元素都是 [...]-shaped Tensor
    r00 = 1.0 - two * (yy + zz)
    r01 = two * (xy - wz)
    r02 = two * (xz + wy)

    r10 = two * (xy + wz)
    r11 = 1.0 - two * (xx + zz)
    r12 = two * (yz - wx)

    r20 = two * (xz - wy)
    r21 = two * (yz + wx)
    r22 = 1.0 - two * (xx + yy)

    # --- 组装旋转矩阵 ---
    R = torch.stack((
            torch.stack((r00, r01, r02), dim=-1),
            torch.stack((r10, r11, r12), dim=-1),
            torch.stack((r20, r21, r22), dim=-1)
        ), dim=-2)          # 最后两维变成 3×3
    return R


# -----------------------------------------------------------
# 2. 旋转向量 v  =  R · v
# -----------------------------------------------------------
@torch.jit.script
def quat_rotate_vector(q: torch.Tensor,          # [..., 4]
                       v: torch.Tensor           # [..., 3]
                      ) -> torch.Tensor:         # [..., 3]
    """
    用四元数 q 旋转任意向量 v。
    q 最后维度是 (w,x,y,z)；q 与 v 的批量维可广播。
    """
    R = quat_to_rotmat(q)                        # [..., 3, 3]
    v_expanded = v.unsqueeze(-1)                 # [..., 3, 1]
    v_rot = torch.matmul(R, v_expanded)          # [..., 3, 1]
    return v_rot.squeeze(-1)                     # [..., 3]


def quaternion_to_normal(w: float, x: float, y: float, z: float) -> np.ndarray:
    """
    将四元数转换为世界坐标系下的法向量 (3D unit vector)。
    """
    # 归一化四元数
    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    
    # 四元数 → 旋转矩阵
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    
    # 基准法向 (0,0,1) 旋转到世界坐标系
    normal = R @ np.array([1.0, 0.0, 0.0])
    normal /= np.linalg.norm(normal)
    return normal

def plot_normal_vector(x: float, y: float, z: float, w: float) -> np.ndarray:
    """
    计算法向量并在 3D 空间中绘制这条向量。
    
    返回值
    ----
    n : np.ndarray
        归一化后的法向量。
    """
    n = quaternion_to_normal(x, y, z, w)
    
    # 绘制
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 画出法向量箭头
    ax.quiver(0, 0, 0, n[0], n[1], n[2], length=1.0, normalize=True)
    
    # 设定坐标轴范围与标签
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Normal Vector from Quaternion')
    
    plt.show()
    return n

# ================= 示例 =================
# 45° 绕 X 轴旋转
# theta = np.deg2rad(45)
# qx, qy, qz = np.sin(theta/2) * np.array([1.0, 0.0, 0.0])
# qw = np.cos(theta/2)
thumb_state = torch.tensor([  0.4620, -0.2666,  0.7706,  0.5642, -0.2396, -0.7435, -0.2]).repeat(2,1)
index_state = torch.tensor([4.9553e-01, -1.6311e-01,  7.5835e-01,  8.4376e-01, -2.8651e-01, 2.6015e-01, -3.7190e-01]).repeat(2,1)
middle_state = torch.tensor([0.5166, -0.2307,  0.7899, -0.3796,  0.3742, -0.0553,  0.84]).repeat(2,1)
thumb_normal_vec = quat_rotate_vector(thumb_state[:,3:7], torch.tensor([0., 1., 0.]))
index_normal_vec = quat_rotate_vector(index_state[:,3:7], torch.tensor([1., 0., 0.]))
middle_normal_vec = quat_rotate_vector(middle_state[:,3:7], torch.tensor([1., 0., 0.]))
normal_vec = torch.stack([thumb_normal_vec, index_normal_vec, middle_normal_vec], dim=-2)
pos = torch.stack([thumb_state[:,:3], index_state[:,:3], middle_state[:,:3]], dim=-2)
rwd = planar_force_closure(pos, normal_vec)
print("rwd:", rwd)
