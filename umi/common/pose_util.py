import numpy as np
import scipy.spatial.transform as st

def pos_rot_to_mat(pos, rot):
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4,4), dtype=pos.dtype)
    mat[...,:3,3] = pos
    mat[...,:3,:3] = rot.as_matrix()
    mat[...,3,3] = 1
    return mat  # [2,4,4]
'''
在您提供的 `pos_rot_to_mat` 函数中，目标是将位置 (`pos`) 和旋转 (`rot`) 转换为一个齐次坐标矩阵 (homogeneous transformation matrix)。这种矩阵在计算机图形、机器人学和许多工程领域中非常重要，因为它可以同时表示三维空间中的旋转和平移。

### 功能解释

1. **输入说明**:
   - `pos`: 位置向量数组，形状为 `[n, 3]`，其中 `n` 是样本数量。
   - `rot`: 由 `st.Rotation` 对象表示的旋转，可提供旋转矩阵。

2. **创建矩阵**:
   - `shape = pos.shape[:-1]` 获取 `pos` 的形状，除去最后一个维度，这里通常是指样本数量。例如，如果 `pos` 是 `[2, 3]`，则 `shape` 会是 `[2]`。
   - `mat = np.zeros(shape + (4,4), dtype=pos.dtype)` 创建一个新的数组，用于存储每个位置和旋转的齐次坐标矩阵。这个矩阵的形状为 `[n, 4, 4]`，其中 `n` 是样本数量。

3. **填充矩阵**:
   - `mat[..., :3, 3] = pos` 将位置向量 `pos` 填充到矩阵的最后一列，除了最后一行外。这部分代表空间中的平移。
   - `mat[..., :3, :3] = rot.as_matrix()` 将由旋转对象 `rot` 计算得到的旋转矩阵填充到矩阵的左上角 3x3 子矩阵中。这部分代表空间中的旋转。
   - `mat[..., 3, 3] = 1` 设置齐次坐标矩阵的右下角元素为 1，这是齐次坐标矩阵的标准形式。

4. **返回值**:
   - 函数返回形状为 `[n, 4, 4]` 的变换矩阵，这里每个矩阵同时包含了对应姿态的旋转和平移信息。

### 用途

这种齐次坐标矩阵非常有用，因为它可以直接用于多种场合：
- **三维图形**：直接用于 OpenGL 或其他图形库中，进行对象的变换。
- **机器人学**：用于描述机器人的关节位置或工具末端执行器的位置和朝向。
- **仿真和分析**：在任何需要精确控制对象位置和方向的场景中，这种矩阵提供了一个简单而强大的工具。

通过转换为齐次坐标矩阵，可以很方便地将多个变换连续应用于一个对象，或是在不同坐标系之间转换，这都得益于矩阵乘法的特性。
'''

def mat_to_pos_rot(mat):
    pos = (mat[...,:3,3].T / mat[...,3,3].T).T
    rot = st.Rotation.from_matrix(mat[...,:3,:3])
    return pos, rot

def pos_rot_to_pose(pos, rot):
    shape = pos.shape[:-1]
    pose = np.zeros(shape+(6,), dtype=pos.dtype)
    pose[...,:3] = pos
    pose[...,3:] = rot.as_rotvec()
    return pose

def pose_to_pos_rot(pose):  # [2,6]
    pos = pose[...,:3]
    rot = st.Rotation.from_rotvec(pose[...,3:])
    return pos, rot  # [2,3] [2]
'''
在处理姿态数据时，通常会涉及位置（position）和旋转（rotation）两个组成部分。在您提供的 pose_to_pos_rot 函数中，输入 pose 是一个形状为 [2, 6] 的数组，这里的处理逻辑和数据分割如下：

解释
数据结构:
pose 数组的每一行代表一个姿态，其中包含6个元素。
前三个元素 (pose[..., :3]) 表示位置（x, y, z 坐标）。
后三个元素 (pose[..., 3:]) 表示旋转，通常以旋转向量的形式表示。
位置（Position）:
pos = pose[..., :3] 这行代码截取每个姿态的前三个元素，即位置信息。结果 pos 的形状为 [2, 3]。
旋转（Rotation）:
rot = st.Rotation.from_rotvec(pose[..., 3:]) 这行代码处理旋转部分。
st.Rotation.from_rotvec() 是 scipy 库中的一个函数，用于从旋转向量创建一个旋转对象。旋转向量是一种表达旋转的方式，其中向量的方向指定旋转轴，而向量的长度指定旋转角度（以弧度为单位）。
返回值:
函数返回两个值：pos（位置）和 rot（旋转对象）。
pos 是一个 [2, 3] 形状的数组，表示两个姿态的位置。
rot 是一个包含两个旋转对象的数组（或类似结构），每个旋转对象对应一个输入姿态的旋转部分。
'''

def pose_to_mat(pose):
    return pos_rot_to_mat(*pose_to_pos_rot(pose))

def mat_to_pose(mat):
    return pos_rot_to_pose(*mat_to_pos_rot(mat))

def transform_pose(tx, pose):
    """
    tx: tx_new_old
    pose: tx_old_obj
    result: tx_new_obj
    """
    pose_mat = pose_to_mat(pose)
    tf_pose_mat = tx @ pose_mat
    tf_pose = mat_to_pose(tf_pose_mat)
    return tf_pose

def transform_point(tx, point):
    return point @ tx[:3,:3].T + tx[:3,3]

def project_point(k, point):  # project_point 函数似乎是用于通过相机内参矩阵将3D点投影到2D坐标的函数。这里是该函数及代码各部分的详细解释：
    x = point @ k.T
    uv = x[...,:2] / x[...,[2]]
    return uv

def apply_delta_pose(pose, delta_pose):
    new_pose = np.zeros_like(pose)

    # simple add for position
    new_pose[:3] = pose[:3] + delta_pose[:3]

    # matrix multiplication for rotation
    rot = st.Rotation.from_rotvec(pose[3:])
    drot = st.Rotation.from_rotvec(delta_pose[3:])
    new_pose[3:] = (drot * rot).as_rotvec()

    return new_pose
'''
这段代码定义了一个名为 `apply_delta_pose` 的函数，用于将一个位姿变化（增量位姿）应用到一个初始位姿上。在机器人学和计算机视觉中，位姿通常包含位置（平移）和方向（旋转）信息。这段代码处理这两部分信息的方式有所不同。下面详细解释一下代码的每一部分：

### 参数解释
- **pose**: 初始位姿，通常是一个包含6个元素的数组，前3个元素表示平移（x, y, z），后3个元素表示旋转（通常使用旋转向量表示）。
- **delta_pose**: 位姿变化，格式与 `pose` 相同，前3个元素是平移的变化量，后3个元素是旋转的变化量。

### 函数的执行过程
1. **初始化新位姿**:
   - `new_pose = np.zeros_like(pose)` 创建一个与 `pose` 形状相同且元素全为0的数组，用来存储计算的结果。

2. **位置更新**:
   - `new_pose[:3] = pose[:3] + delta_pose[:3]` 直接将位置变化量加到初始位置上，更新位姿的位置部分。

3. **旋转更新**:
   - 使用 `scipy.spatial.transform.Rotation` 模块（这里简写为 `st`），这是一个处理3D旋转的强大工具。
     - `rot = st.Rotation.from_rotvec(pose[3:])` 将初始位姿的旋转向量转换为一个旋转对象。
     - `drot = st.Rotation.from_rotvec(delta_pose[3:])` 将旋转变化量也转换为一个旋转对象。
   - `new_pose[3:] = (drot * rot).as_rotvec()` 将旋转变化量应用到初始旋转上。这里使用的是旋转的组合（乘法），意味着先应用 `rot`，然后应用 `drot`。结果旋转再转换为旋转向量格式存储在 `new_pose` 中。

### 返回值
- **new_pose**: 更新后的位姿，包括新的位置和旋转信息。

### 总结
这个函数用于更新一个3D位姿，通过加上一个平移增量和组合一个旋转增量。这种类型的操作在机器人路径规划、动画、游戏开发以及任何涉及3D空间运动的场景中非常重要和常见。通过这种方式可以方便地计算和更新物体的位置和方向。
'''
def normalize(vec, tol=1e-7):
    return vec / np.maximum(np.linalg.norm(vec), tol)

def rot_from_directions(from_vec, to_vec):
    from_vec = normalize(from_vec)
    to_vec = normalize(to_vec)
    axis = np.cross(from_vec, to_vec)
    axis = normalize(axis)
    angle = np.arccos(np.dot(from_vec, to_vec))
    rotvec = axis * angle
    rot = st.Rotation.from_rotvec(rotvec)
    return rot

def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    out = (vec.T / norm).T
    return out

def rot6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return out

'''
mat_to_rot6d 函数该mat_to_rot6d函数将 3x3 旋转矩阵转换为 6 维旋转向量。此函数通常用于计算机视觉和机器人技术，以更紧凑的形式表示旋转。
'''
def mat_to_rot6d(mat):
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out

def mat_to_pose10d(mat):
    pos = mat[...,:3,3]
    rotmat = mat[...,:3,:3]
    d6 = mat_to_rot6d(rotmat)
    d10 = np.concatenate([pos, d6], axis=-1)
    return d10

def pose10d_to_mat(d10):
    pos = d10[...,:3]
    d6 = d10[...,3:]
    rotmat = rot6d_to_mat(d6)
    out = np.zeros(d10.shape[:-1]+(4,4), dtype=d10.dtype)
    out[...,:3,:3] = rotmat
    out[...,:3,3] = pos
    out[...,3,3] = 1
    return out
