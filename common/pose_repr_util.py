import numpy as np

'''
这段代码定义的 compute_relative_pose 函数用于计算位置和旋转的相对或增量姿态，具有正向和反向计算的能力。函数接受多个参数来处理不同的情况：
参数说明
pos: 当前位置向量或矩阵。
rot: 当前旋转表示，具体形式取决于 rot_transformer_to_mat 函数的输入要求。
base_pos: 基准位置向量。
base_rot_mat: 基准旋转矩阵。
rot_transformer_to_mat: 转换 rot 到矩阵形式的函数或类的实例。
rot_transformer_to_target: 将旋转矩阵转换回 rot 的原始表示形式的函数或类的实例。
backward: 布尔值，指示计算是正向还是反向。
delta: 布尔值，指示输出是相对差值还是累积值。
函数逻辑
正向计算 (backward=False)
非增量模式 (delta=False):
位置：如果 base_pos 为空，则输出位置即 pos；否则输出位置为 pos - base_pos。
旋转：计算 rot 的矩阵形式，与基准旋转矩阵的逆矩阵相乘，再将结果转换回目标旋转形式。
增量模式 (delta=True):
位置：计算 pos 和 base_pos 的差分。
旋转：将 rot 转换为矩阵形式，并与基准旋转矩阵合并，计算差分旋转，最后转换回目标旋转形式。
反向计算 (backward=True)
非增量模式 (delta=False):
位置：如果 base_pos 为空，则输出位置即 pos；否则输出位置为 pos + base_pos。
旋转：将 rot 从目标形式转换为矩阵形式，与基准旋转矩阵相乘，然后将结果转换回原始旋转形式。
增量模式 (delta=True):
位置：累积 pos 的变化并加上 base_pos。
旋转：从目标旋转形式转换 rot，通过累积方式应用旋转变化，最后将结果转换回矩阵形式。
'''
def compute_relative_pose(pos, rot, base_pos, base_rot_mat,
                          rot_transformer_to_mat,
                          rot_transformer_to_target,
                          backward=False,
                          delta=False):
    if not backward:
        # forward pass
        if not delta:
            output_pos = pos if base_pos is None else pos - base_pos
            output_rot = rot_transformer_to_target.forward(
                rot_transformer_to_mat.forward(rot) @ np.linalg.inv(base_rot_mat))
            return output_pos, output_rot
        else:
            all_pos = np.concatenate([base_pos[None,...], pos], axis=0)
            output_pos = np.diff(all_pos, axis=0)
            
            rot_mat = rot_transformer_to_mat.forward(rot)
            all_rot_mat = np.concatenate([base_rot_mat[None,...], rot_mat], axis=0)
            prev_rot = np.linalg.inv(all_rot_mat[:-1])
            curr_rot = all_rot_mat[1:]
            rot = np.matmul(curr_rot, prev_rot)
            output_rot = rot_transformer_to_target.forward(rot)
            return output_pos, output_rot
            
    else:
        # backward pass
        if not delta:
            output_pos = pos if base_pos is None else pos + base_pos
            output_rot = rot_transformer_to_mat.inverse(
                rot_transformer_to_target.inverse(rot) @ base_rot_mat)
            return output_pos, output_rot
        else:
            output_pos = np.cumsum(pos, axis=0) + base_pos
            
            rot_mat = rot_transformer_to_target.inverse(rot)
            output_rot_mat = np.zeros_like(rot_mat)
            curr_rot = base_rot_mat
            for i in range(len(rot_mat)):
                curr_rot = rot_mat[i] @ curr_rot
                output_rot_mat[i] = curr_rot
            output_rot = rot_transformer_to_mat.inverse(rot)
            return output_pos, output_rot

'''
这段代码定义了一个函数 convert_pose_mat_rep，该函数对输入的姿态矩阵进行不同的变换，以适应不同的表示形式（如绝对坐标、相对坐标等）。此函数接受四个参数：

pose_mat: 输入的姿态矩阵。
base_pose_mat: 基准姿态矩阵，用于计算相对或增量姿态。
pose_rep: 表示姿态的方式。支持 'abs'（绝对）、'rel'（相对，有bug的旧实现）、'relative'（正确的相对姿态）、'delta'（差分或增量姿态）。
backward: 布尔值，指定是否反向变换，通常用于评估阶段。
绝对姿态 'abs':
直接返回 pose_mat（无论 backward 是否为真）。
相对姿态 'rel'（有bug的旧实现）:
正向：计算位置差和旋转的逆，将结果保存在新的姿态矩阵中。
反向：计算位置和旋转的和，将结果保存在新的姿态矩阵中。
相对姿态 'relative'（正确实现）:
正向：通过预先计算基姿态的逆矩阵，然后与输入姿态矩阵相乘。
反向：直接将基姿态矩阵与输入姿态矩阵相乘。
增量姿态 'delta':
正向：计算位置和旋转的差分。
反向：通过累积输入姿态的变化来重建位置和旋转
'''
def convert_pose_mat_rep(pose_mat, base_pose_mat, pose_rep='abs', backward=False):  # base_pose_mat-(4,4) pose_mat-(2,4,4) pose_rep-'relative'
    if not backward:
        # training transform
        if pose_rep == 'abs':
            return pose_mat
        elif pose_rep == 'rel':
            # legacy buggy implementation
            # for compatibility
            pos = pose_mat[...,:3,3] - base_pose_mat[:3,3]
            rot = pose_mat[...,:3,:3] @ np.linalg.inv(base_pose_mat[:3,:3])
            out = np.copy(pose_mat)
            out[...,:3,:3] = rot
            out[...,:3,3] = pos
            return out
        elif pose_rep == 'relative':  # use it
            out = np.linalg.inv(base_pose_mat) @ pose_mat
            return out  # (2,4,4)
        elif pose_rep == 'delta':
            all_pos = np.concatenate([base_pose_mat[None,:3,3], pose_mat[...,:3,3]], axis=0)
            out_pos = np.diff(all_pos, axis=0)
            
            all_rot_mat = np.concatenate([base_pose_mat[None,:3,:3], pose_mat[...,:3,:3]], axis=0)
            prev_rot = np.linalg.inv(all_rot_mat[:-1])
            curr_rot = all_rot_mat[1:]
            out_rot = np.matmul(curr_rot, prev_rot)
            
            out = np.copy(pose_mat)
            out[...,:3,:3] = out_rot
            out[...,:3,3] = out_pos
            return out
        else:
            raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")

    else:
        # eval transform
        if pose_rep == 'abs':
            return pose_mat
        elif pose_rep == 'rel':
            # legacy buggy implementation
            # for compatibility
            pos = pose_mat[...,:3,3] + base_pose_mat[:3,3]
            rot = pose_mat[...,:3,:3] @ base_pose_mat[:3,:3]
            out = np.copy(pose_mat)
            out[...,:3,:3] = rot
            out[...,:3,3] = pos
            return out
        elif pose_rep == 'relative':
            out = base_pose_mat @ pose_mat
            return out
        elif pose_rep == 'delta':
            output_pos = np.cumsum(pose_mat[...,:3,3], axis=0) + base_pose_mat[:3,3]
            
            output_rot_mat = np.zeros_like(pose_mat[...,:3,:3])
            curr_rot = base_pose_mat[:3,:3]
            for i in range(len(pose_mat)):
                curr_rot = pose_mat[i,:3,:3] @ curr_rot
                output_rot_mat[i] = curr_rot
            
            out = np.copy(pose_mat)
            out[...,:3,:3] = output_rot_mat
            out[...,:3,3] = output_pos
            return out
        else:
            raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")
'''
在您提供的代码片段中，`pose_rep == 'relative'` 表示正在进行的操作是计算一个相对于基准位置的姿态矩阵。这是一个在机器人学、计算机视觉和相关领域中常见的操作，用于描述一个物体或观察点相对于一个参照物（基础或基准）的位置和方向。
### 解释这段代码的功能和逻辑：
- **`base_pose_mat`**：这是基准姿态的齐次坐标矩阵。它是一个 4x4 的矩阵，描述了一个从参考坐标系（比如世界坐标系或机器人的基座坐标系）到某个特定坐标系（比如机器人末端执行器或某个物体）的变换。
- **`pose_mat`**：这是另一个齐次坐标矩阵，描述了从相同的参考坐标系到另一个物体或观察点的变换。
- **计算相对姿态**：
  - `np.linalg.inv(base_pose_mat)`：计算基准姿态矩阵的逆。逆矩阵用于将坐标从基准姿态坐标系变换回参考坐标系。
  - `@`：这是 Python 中的矩阵乘法操作符，用于矩阵的乘法。
  - `np.linalg.inv(base_pose_mat) @ pose_mat`：这个操作首先将任何在基准姿态坐标系中的坐标变换回参考坐标系，然后再通过 `pose_mat` 变换到另一个姿态的坐标系。这个结果是一个新的齐次坐标矩阵，描述了从基准姿态直接到目标姿态的相对变换。
### 结果解释：
这段代码的结果是一个描述相对变换的齐次坐标矩阵 `out`。这意味着，如果你有一个点或向量在基准姿态下的坐标，你可以使用矩阵 `out` 来计算它在目标姿态下的坐标。这种相对姿态计算在多机器人系统、机器人手眼协调、以及场景理解等多个领域有广泛的应用。
### 应用：
- 在机器人中，这可以用于计算工具坐标系中的传感器数据，便于理解和操作。
- 在增强现实（AR）和虚拟现实（VR）中，相对姿态可以帮助正确渲染物体相对于用户视角的位置。
'''