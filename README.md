# umi_data_deal1

## 构建环境：
```console
sudo apt-get update

sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf　（如果ｕnbuntu有系统问题，可以创建docker，这里偷懒了，没有用docker）

conda env create -f conda_environment.yaml　（这一步如果太慢，就按作者的推荐来使用Miniforge）

conda activate umi_data
```
## 得到原始采集视频做预处理（0-6步follow umi）：
```console
python run_slam_pipeline.py　"umi_data_deal/degug_session"
```
0：所有示例数据目录结构整理：每个视频都要单独处理，用ExifTool 工具包，提取每个视频的相机序列号+拍摄时间，作为文件夹的名称，如demos文件夹所示．

1：提取gopro惯性测量单元数据(imu)，提取方式是拉的docker镜像，直接使用的外部仓库：GitHub - urbste/OpenImuCameraCalibrator: Camera calibration tool，提取结果保存在imu_data.json文件中，总共提取了6种数据：GoPro Tags．

2：处理mapping地图视频数据，并生成地图。主要是用Docker来运行外部库ORB_SLAM3(Simultaneous Localization and Mapping，即同时定位与地图构建)

输入：imu_data.json 和 原MP4视频

输出：mapping_camera_trajectory.csv这是SLAM系统生成的相机轨迹文件，通常包含了相机在空间中的位置和方向信息．

3：mapping地图视频生成的轨迹信息，

输入：原始mp4视频、上一步生成的map_atlas.osa、imu_data.json

输出：相机轨迹信息camera_trajectory.csv．

4：批量生成任务演示数据的轨迹信息．

5：SLAM标签校准,夹爪范围的校准．

6：
相机坐标系到夹爪尖坐标系的转换

加载机器人夹爪的校准数据

提取每个视频的元数据

视频文件进行分组

识别每个视频中的夹爪硬件ID

确定每个摄像头在演示中的左右位置

准备模型训练的数据集


## 7.数据转换+存储hdf5
下载数据到目录中：https://drive.google.com/drive/folders/1U3B7NzntI2jEL2IK7irBgPji_gc30Nkz?usp=drive_link，

degug_session是已经执行完python run_slam_pipeline.py　"umi_data_deal/degug_session"的数据了，但是只放了4个视频（1map,1gripper标定，2训练数据），如果想大量训练可以来我这里拷贝，比较大。

接下来单执行07_generate_replay_buffer.py做转换存储就可以了，存储部分暂时不用改，现在写的够用，请大佬们重写转换motion_convert（），

2个问题：

1-pose到joint的转换，里面的constants都是从mobile aloha粘的。（转换方法求大佬们重写）

2-gripper width到aloha的转换，在转换前，可以先打开plot_gripper_width/2看下实际的宽度变化，再看我写的方法。（转换方法求大佬们优化，注释部分是想用滑动窗口，但是有顾虑没用）


```console
python scripts_slam_pipeline/07_generate_replay_buffer.py -o degug_session/dataset.zarr.zip degug_session**
```
*7:所以重点看这个，数据转换＋存储，07_generate_replay_buffer.py*
