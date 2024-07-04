"""
python scripts_slam_pipeline/02_create_map.py -i data/dataset/raw_videos/demos/mapping
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import click
import subprocess
import numpy as np
import cv2
from umi.common.cv_util import draw_predefined_mask

'''
处理mapping地图视频数据，并生成地图。主要是用Docker来运行外部库ORB_SLAM3(Simultaneous Localization and Mapping，即同时定位与地图构建)系统
输入：上文《4.2.1 01_extract_gopro_imu.py：提取gopro惯性测量单元数据(imu)》中的的imu_data.json 和 原MP4视频
输出：
mapping_camera_trajectory.csv
这是SLAM系统生成的相机轨迹文件，通常包含了相机在空间中的位置和方向信息
1) frame_idx：帧索引，表示该帧在视频中的顺序编号
2) timestamp: 时间戳，表示该帧的拍摄时间，通常是以秒或毫秒为单位
3) state：状态字段，通常用于表示SLAM系统的当前状态，例如是否初始化、是否丢失、是否关键帧等
4) is_lost：是否丢失标志，表示该帧是否丢失或无法追踪
5) is_keyframe：是否关键帧标志，表示该帧是否被选为关键帧，关键帧是SLAM中用于地图构建的重要帧
6) x、y、z：相机在三维空间中的平移坐标
7) q_x、q_y、q_z、q_w：相机姿态的四元数表示，用于描述相机的旋转。其中， q_x、q_y、q_z 分别表示轴向量的xyz分量，q_w 是四元数的第一个元素，表示旋转的轴向量
map_atlas.osa
这个文检是个二进制文件，无法打开，一般是map_atlas.osa 文件是一个专用的 ORB_SLAM3 地图文件格式，它不是公开的标准格式，所以没有具体的公开文档说明其内部结构。
ORB_SLAM3 是用c++写的，所以，要看懂，需要掌握slam和c++编程。
不过，通常这类 SLAM系统的地图文件会包含以下类型的数据：
a) 关键帧数据：关键帧是 SLAM 系统中用于定位和地图构建的参考框架。它们通常包含了位置、姿态信息以及与之相关的地图点的标识符。
b) 地图点数据：地图点是 SLAM 系统中在环境中固定的特征点，如角点或边缘。它们在多个关键帧中被观测和记录，以帮助建立一个连续的位置估计。
c) 连接关系：关键帧之间的连接关系表示了时间上的连续性，这对于递归定位和构建地图至关重要。
d) 其他参数和信息：这可能包括与特定地图创建过程相关的参数，如滤波器的配置、重定位参数等
'''

@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for mapping video')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('-d', '--docker_image', default="chicheng/orb_slam3:latest")
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="pull docker image from docker hub")
@click.option('-nm', '--no_mask', is_flag=True, default=False, help="Whether to mask out gripper and mirrors. Set if map is created with bare GoPro no on gripper.")
def main(input_dir, map_path, docker_image, no_docker_pull, no_mask):
    video_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    for fn in ['raw_video.mp4', 'imu_data.json']:
        assert video_dir.joinpath(fn).is_file()

    if map_path is None:
        map_path = video_dir.joinpath('map_atlas.osa')
    else:
        map_path = pathlib.Path(os.path.expanduser(map_path)).absolute()
    map_path.parent.mkdir(parents=True, exist_ok=True)

    # pull docker
    if not no_docker_pull:
        print(f"Pulling docker image {docker_image}")
        cmd = [
            'docker',
            'pull',
            docker_image
        ]
        p = subprocess.run(cmd)
        if p.returncode != 0:
            print("Docker pull failed!")
            exit(1)

    mount_target = pathlib.Path('/data')
    csv_path = mount_target.joinpath('mapping_camera_trajectory.csv')
    video_path = mount_target.joinpath('raw_video.mp4')
    json_path = mount_target.joinpath('imu_data.json')
    mask_path = mount_target.joinpath('slam_mask.png')
    if not no_mask:
        mask_write_path = video_dir.joinpath('slam_mask.png')
        slam_mask = np.zeros((2028, 2704), dtype=np.uint8)
        slam_mask = draw_predefined_mask(
            slam_mask, color=255, mirror=True, gripper=False, finger=True)
        cv2.imwrite(str(mask_write_path.absolute()), slam_mask)

    map_mount_source = pathlib.Path(map_path)
    map_mount_target = pathlib.Path('/map').joinpath(map_mount_source.name)

    # run SLAM
    cmd = [
        'docker',
        'run',
        '--rm', # delete after finish
        '--volume', str(video_dir) + ':' + '/data',
        '--volume', str(map_mount_source.parent) + ':' + str(map_mount_target.parent),
        docker_image,
        '/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam',
        '--vocabulary', '/ORB_SLAM3/Vocabulary/ORBvoc.txt',
        '--setting', '/ORB_SLAM3/Examples/Monocular-Inertial/gopro10_maxlens_fisheye_setting_v1_720.yaml',
        '--input_video', str(video_path),
        '--input_imu_json', str(json_path),
        '--output_trajectory_csv', str(csv_path),
        '--save_map', str(map_mount_target)
    ]
    if not no_mask:
        cmd.extend([
            '--mask_img', str(mask_path)
        ])

    stdout_path = video_dir.joinpath('slam_stdout.txt')
    stderr_path = video_dir.joinpath('slam_stderr.txt')

    result = subprocess.run(
        cmd,
        cwd=str(video_dir),
        stdout=stdout_path.open('w'),
        stderr=stderr_path.open('w')
    )
    print('Done!')

if __name__ == "__main__":
    main()
