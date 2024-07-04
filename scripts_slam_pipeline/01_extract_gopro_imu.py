"""
This script extracts GoPro Inertial Measurement Unit (IMU) data using an external Docker image.
The extracted data includes six types of GoPro tags: ACCL (Accelerometer), GYRO (Gyroscope), GPS5 (Global Positioning System),
CORI (Camera Orientation), IORI (Image Orientation), and GRAV (Gravity Vector). The results are saved in imu_data.json files.

Usage:
    python scripts_slam_pipeline/01_extract_gopro_imu.py data/dataset
"""

import sys
import os

# Ensure the necessary paths are added before importing other modules
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import click
import subprocess
import multiprocessing
import concurrent.futures
from tqdm import tqdm

@click.command()
@click.option('-d', '--docker_image', default="chicheng/openicc:latest", help="Docker image to use for extraction")
@click.option('-n', '--num_workers', type=int, default=None, help="Number of worker threads")
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="Disable Docker image pulling")
@click.argument('session_dir', nargs=-1)

############################################################################################
# ""这步用于提取gopro惯性测量单元数据(imu)，提取方式是拉的docker镜像，直接使用的外部仓库：GitHub - urbste/OpenImuCameraCalibrator: Camera calibration tool
# 且是C++写的，直接看提取结果，保存在imu_data.json文件中，总共提取了6种数据：GoPro Tags，这六种数据分别如下所示
# 1. ACCL (Accelerometer)加速度计:
# 加速度计测量物体在三个方向上的加速度，通常分别是 X 轴、Y 轴和 Z 轴。这些数据用于检测物体的运动和方向变化，共有四个值
# value：代表了三个轴的加速度值：x 轴、y 轴和 z 轴。单位：m/s2
# cts: 采样的时间戳，可能表示这个数据点是在第x个采样时钟周期中采集的
# data: 采样的日期和时间
# temperature: 采样的温度
# {
# "value":[8.37410071942446,0.5875299760191847,4.9352517985611515],
# "cts":78.018,
# "date":"2024-01-10T18:54:47.109Z",
# "temperature [°C]":51.544921875
# }

# 2.GYRO (Gyroscope)陀螺仪
# 陀螺仪测量物体在三个方向上的角速度，即物体围绕每个轴旋转的速度。陀螺仪用于确定物体的姿态和运动状态，对于检测旋转和倾斜非常有效。
# value：代表了三个轴的角速度：x 轴、y 轴和 z 轴，单位：rad/s
# cts: 采样的时间戳
# data: 采样的日期和时间
# temperature: 采样的温度
# {
# "value":[0.06496272630457935,0.0724174653887114,-0.027689030883919063],
# "cts":78.018,
# "date":"2024-01-10T18:54:47.109Z",
# "temperature [°C]":51.544921875
# }

# ３．GPS5 (Global Positioning System)全球定位系统
# GPS 传感器提供位置数据，包括经度、纬度、高度以及速度。GPS 数据用于定位和导航

# ４．CORI( Camera Orientation)相机姿态
# 通常用于图像处理和计算机视觉领域，用于描述图像在三维空间中的方向和位置。
# value：代表了三个轴的加速度值：x 轴、y 轴和 z 轴。
# cts: 采样的时间戳
# data: 采样的日期和时间
# temperature: 采样的温度
# {
# "value":[0.999969481490524,0.002044740134891812,0.0016174810022278512,-0.0003662221137119663],
# "cts":80.255,
# "date":"2024-01-10T18:54:47.109Z"
# }

# ５．IORI（Image Orientation）图像姿态
# ６．GRAV (Gravity Vector)重力向量""
############################################################################################

def main(docker_image, num_workers, no_docker_pull, session_dir):
    """
    Main function to extract IMU data from GoPro videos in the specified session directories.

    Args:
        docker_image (str): Docker image to use for extraction.
        num_workers (int): Number of worker threads.
        no_docker_pull (bool): Flag to disable Docker image pulling.
        session_dir (list): List of session directories containing raw videos.

    Returns:
        None
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    # Pull Docker image unless no_docker_pull flag is set
    if not no_docker_pull:
        print(f"Pulling Docker image {docker_image}")
        cmd = ['docker', 'pull', docker_image]
        p = subprocess.run(cmd)
        if p.returncode != 0:
            print("Docker pull failed!")
            exit(1)

    for session in session_dir:
        input_dir = pathlib.Path(os.path.expanduser(session)).joinpath('demos')
        input_video_dirs = [x.parent for x in input_dir.glob('*/raw_video.mp4')]
        print(f'Found {len(input_video_dirs)} video directories')

        with tqdm(total=len(input_video_dirs)) as pbar:
            # Use ThreadPoolExecutor to parallelize the extraction process
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = set()
                for video_dir in tqdm(input_video_dirs):
                    video_dir = video_dir.absolute()
                    if video_dir.joinpath('imu_data.json').is_file():
                        print(f"imu_data.json already exists, skipping {video_dir.name}")
                        continue

                    mount_target = pathlib.Path('/data')
                    video_path = mount_target.joinpath('raw_video.mp4')
                    json_path = mount_target.joinpath('imu_data.json')

                    # Command to run IMU extractor
                    cmd = [
                        'docker', 'run', '--rm', '--volume', str(video_dir) + ':/data',
                        docker_image, 'node', '/OpenImuCameraCalibrator/javascript/extract_metadata_single.js',
                        str(video_path), str(json_path)
                    ]

                    stdout_path = video_dir.joinpath('extract_gopro_imu_stdout.txt')
                    stderr_path = video_dir.joinpath('extract_gopro_imu_stderr.txt')

                    if len(futures) >= num_workers:
                        # Limit number of inflight tasks
                        completed, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                        pbar.update(len(completed))

                    futures.add(executor.submit(
                        lambda x, stdo, stde: subprocess.run(
                            x, cwd=str(video_dir), stdout=stdo.open('w'), stderr=stde.open('w')), 
                        cmd, stdout_path, stderr_path))

                # Wait for all tasks to complete
                completed, futures = concurrent.futures.wait(futures)
                pbar.update(len(completed))

        print("Done!")
        # print([x.result() for x in completed])

if __name__ == "__main__":
    main()
