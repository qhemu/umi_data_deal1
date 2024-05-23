from typing import Optional
import numpy as np
import random
import scipy.interpolate as si
import scipy.spatial.transform as st
from diffusion_policy.common.replay_buffer import ReplayBuffer

def get_val_mask(n_episodes, val_ratio, seed=0):  # 0.05 5 42
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train 至少有 1 集用于验证，至少 1 集用于训练
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)  # 1
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


class SequenceSampler:
    def __init__(self,
        shape_meta: dict,
        replay_buffer: ReplayBuffer,
        rgb_keys: list,  # ['camera0_rgb']
        lowdim_keys: list,  # ['robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width', 'robot0_demo_end_pose', 'robot0_demo_start_pose']
        key_horizon: dict,  # {'action': 16, 'camera0_rgb': 2, 'robot0_demo_end_pose': 2, 'robot0_demo_start_pose': 2, 'robot0_eef_pos': 2, 'robot0_eef_rot_axis_angle': 2, 'robot0_eef_rot_axis_angle_wrt_start': 2, 'robot0_gripper_width': 2}
        key_latency_steps: dict,  # {'action': 0, 'camera0_rgb': 0, 'robot0_demo_end_pose': 0.0, 'robot0_demo_start_pose': 0.0, 'robot0_eef_pos': 0.0, 'robot0_eef_rot_axis_angle': 0.0, 'robot0_eef_rot_axis_angle_wrt_start': 0.0, 'robot0_gripper_width': 0.0}
        key_down_sample_steps: dict,  # {'action': 3, 'camera0_rgb': 3, 'robot0_demo_end_pose': 3, 'robot0_demo_start_pose': 3, 'robot0_eef_pos': 3, 'robot0_eef_rot_axis_angle': 3, 'robot0_eef_rot_axis_angle_wrt_start': 3, 'robot0_gripper_width': 3}
        episode_mask: Optional[np.ndarray]=None,  # [ True False False False False]
        action_padding: bool=False,
        repeat_frame_prob: float=0.0,
        max_duration: Optional[float]=None
    ):
        episode_ends = replay_buffer.episode_ends[:]  # 468,932,1302,1710,2315

        # load gripper_width
        gripper_width = replay_buffer['robot0_gripper_width'][:, 0]  # 2315
        gripper_width_threshold = 0.08
        self.repeat_frame_prob = repeat_frame_prob

        # create indices, including (current_idx, start_idx, end_idx) 创建索引，包括（current_idx、start_idx、end_idx）
        indices = list()
        for i in range(len(episode_ends)):
            before_first_grasp = True # initialize for each episode 为每一集初始化
            if episode_mask is not None and not episode_mask[i]:
                # skip episode
                continue
            start_idx = 0 if i == 0 else episode_ends[i-1]  # 468
            end_idx = episode_ends[i]  # 932
            if max_duration is not None:
                end_idx = min(end_idx, max_duration * 60)
            for current_idx in range(start_idx, end_idx):
                if not action_padding and end_idx < current_idx + (key_horizon['action'] - 1) * key_down_sample_steps['action'] + 1:
                    continue
                if gripper_width[current_idx] < gripper_width_threshold:
                    before_first_grasp = False
                indices.append((current_idx, start_idx, end_idx, before_first_grasp))
        
        # load low_dim to memory and keep rgb as compressed zarr array 将 low_dim 加载到内存并将 rgb 保留为压缩的 zarr 数组
        self.replay_buffer = dict()
        self.num_robot = 0
        # lowdim_keys　5:'robot0_eef_pos','robot0_eef_rot_axis_angle'，'robot0_gripper_width'，'robot0_demo_end_pose','robot0_demo_start_pose'
        for key in lowdim_keys:
            if key.endswith('eef_pos'):
                self.num_robot += 1

            if key.endswith('pos_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                self.replay_buffer[key] = replay_buffer[key[:-4]][:, list(axis)]
            elif key.endswith('quat_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                # HACK for hybrid abs/relative proprioception
                rot_in = replay_buffer[key[:-4]][:]
                rot_out = st.Rotation.from_quat(rot_in).as_euler('XYZ')
                self.replay_buffer[key] = rot_out[:, list(axis)]
            elif key.endswith('axis_angle_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                rot_in = replay_buffer[key[:-4]][:]
                rot_out = st.Rotation.from_rotvec(rot_in).as_euler('XYZ')
                self.replay_buffer[key] = rot_out[:, list(axis)]
            else:
                self.replay_buffer[key] = replay_buffer[key][:]
        for key in rgb_keys:
            self.replay_buffer[key] = replay_buffer[key]
        
        
        if 'action' in replay_buffer:
            self.replay_buffer['action'] = replay_buffer['action'][:]
        else:
            # construct action (concatenation of [eef_pos, eef_rot, gripper_width])
            actions = list()
            for robot_idx in range(self.num_robot):
                for cat in ['eef_pos', 'eef_rot_axis_angle', 'gripper_width']:
                    key = f'robot{robot_idx}_{cat}'
                    if key in self.replay_buffer:
                        actions.append(self.replay_buffer[key])
            self.replay_buffer['action'] = np.concatenate(actions, axis=-1)

        self.action_padding = action_padding
        self.indices = indices
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        
        self.ignore_rgb_is_applied = False # speed up the interation when getting normalizaer

    def __len__(self):
        return len(self.indices)
    
    def sample_sequence(self, idx):
        current_idx, start_idx, end_idx, before_first_grasp = self.indices[idx]

        result = dict()

        obs_keys = self.rgb_keys + self.lowdim_keys
        if self.ignore_rgb_is_applied:
            obs_keys = self.lowdim_keys

        # observation
        for key in obs_keys:
            input_arr = self.replay_buffer[key]  # <zarr.core.Array '/data/camera0_rgb' (2315, 224, 224, 3) uint8>
            this_horizon = self.key_horizon[key]  # 2
            this_latency_steps = self.key_latency_steps[key]  # 0
            this_downsample_steps = self.key_down_sample_steps[key]  # 3
            
            if key in self.rgb_keys:
                assert this_latency_steps == 0
                num_valid = min(this_horizon, (current_idx - start_idx) // this_downsample_steps + 1)  # 2
                slice_start = current_idx - (num_valid - 1) * this_downsample_steps  # 2122

                output = input_arr[slice_start: current_idx + 1: this_downsample_steps]  # [2,224,224,3]
                assert output.shape[0] == num_valid
                
                # solve padding
                if output.shape[0] < this_horizon:
                    padding = np.repeat(output[:1], this_horizon - output.shape[0], axis=0)
                    output = np.concatenate([padding, output], axis=0)
            else:
                idx_with_latency = np.array(
                    [current_idx - idx * this_downsample_steps + this_latency_steps for idx in range(this_horizon)],
                    dtype=np.float32)
                idx_with_latency = idx_with_latency[::-1]
                idx_with_latency = np.clip(idx_with_latency, start_idx, end_idx - 1)
                interpolation_start = max(int(idx_with_latency[0]) - 5, start_idx)
                interpolation_end = min(int(idx_with_latency[-1]) + 2 + 5, end_idx)

                if 'rot' in key:
                    # rotation
                    rot_preprocess, rot_postprocess = None, None
                    if key.endswith('quat'):
                        rot_preprocess = st.Rotation.from_quat
                        rot_postprocess = st.Rotation.as_quat
                    elif key.endswith('axis_angle'):
                        rot_preprocess = st.Rotation.from_rotvec
                        rot_postprocess = st.Rotation.as_rotvec
                    else:
                        raise NotImplementedError
                    slerp = st.Slerp(
                        times=np.arange(interpolation_start, interpolation_end),
                        rotations=rot_preprocess(input_arr[interpolation_start: interpolation_end]))
                    output = rot_postprocess(slerp(idx_with_latency))
                else:
                    interp = si.interp1d(
                        x=np.arange(interpolation_start, interpolation_end),
                        y=input_arr[interpolation_start: interpolation_end],
                        axis=0, assume_sorted=True)
                    output = interp(idx_with_latency)
                
            result[key] = output

        # repeat frame before first grasp
        if self.repeat_frame_prob != 0.0:
            if before_first_grasp and random.random() < self.repeat_frame_prob:
                for key in obs_keys:
                    result[key][:-1] = result[key][-1:]

        # aciton
        input_arr = self.replay_buffer['action']  # [2315,7]
        action_horizon = self.key_horizon['action']  # 16
        action_latency_steps = self.key_latency_steps['action']  # 0
        assert action_latency_steps == 0
        action_down_sample_steps = self.key_down_sample_steps['action']  # 3
        slice_end = min(end_idx, current_idx + (action_horizon - 1) * action_down_sample_steps + 1)  # 2171
        output = input_arr[current_idx: slice_end: action_down_sample_steps]  # (16,7)
        # solve padding
        if not self.action_padding:
            assert output.shape[0] == action_horizon
        elif output.shape[0] < action_horizon:
            padding = np.repeat(output[-1:], action_horizon - output.shape[0], axis=0)
            output = np.concatenate([output, padding], axis=0)
        result['action'] = output

        return result
    
    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply