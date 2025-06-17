import numpy as np
import matplotlib.pyplot as plt


# ---------- 基础：四元数 → 旋转矩阵 ----------
def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Isaac Sim 四元数 [x, y, z, w] → 3×3 旋转矩阵（假设已归一化）"""
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)]
    ])


# ---------- 新增：四元数 → 欧拉角 (XYZ / roll-pitch-yaw) ----------
def quat_to_euler_xyz(q: np.ndarray, *, degrees: bool = True):
    """
    返回 (roll, pitch, yaw)；默认用度制。
    公式来源：标准旋转矩阵推导，适用 pitch ∈ (-90°, +90°) 主值区。
    """
    R = quat_to_rot_matrix(q)
    # pitch = asin(-R[2,0])
    pitch = np.arcsin(-R[2, 0])
    # roll  = atan2(R[2,1], R[2,2])
    roll  = np.arctan2(R[2, 1], R[2, 2])
    # yaw   = atan2(R[1,0], R[0,0])
    yaw   = np.arctan2(R[1, 0], R[0, 0])

    if degrees:
        return np.degrees([roll, pitch, yaw])
    return roll, pitch, yaw


# ---------- 只画局部 Z 轴 ----------
def plot_z_axis(q: np.ndarray):
    R = quat_to_rot_matrix(q)
    z_world = R @ np.array([0., 0., 1.])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(0, 0, 0, *z_world, length=1.0, arrow_length_ratio=0.1)

    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Rotated local Z-axis')
    ax.view_init(elev=20, azim=135)
    plt.show()


# ---------- Demo ----------
if __name__ == "__main__":
    # 示例：绕 Y 轴旋转 45°
    angle = np.deg2rad(45)
    # q_demo = np.array([-1.6566e-01, -6.7212e-08,
    #       1.4373e-06,  9.8618e-01])
    q_demo = np.array([-1.6566e-01,  2.4822e-07,
          3.8773e-07,  9.8618e-01])

    # 打印欧拉角
    roll, pitch, yaw = quat_to_euler_xyz(q_demo)
    print(f"Euler XYZ (deg): roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}")

    # 画 Z 轴
    plot_z_axis(q_demo)
