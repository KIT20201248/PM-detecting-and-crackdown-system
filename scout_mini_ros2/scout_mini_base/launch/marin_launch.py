"""
- Scout Mini, Velodyne LiDAR, Intel D455 카메라, GenZ-ICP 오도메트리 노드를 동시에 실행하는 ROS 2 런치 파일
- 각 하드웨어/알고리즘별 개별 런치 파일을 IncludeLaunchDescription으로 불러와 병렬 구동
- RViz2를 함께 실행해 센서 데이터, 로봇 상태, 스캔매칭 기반 오도메트리 결과를 시각화
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
import xacro

# 네가지 런치 파일 동시 실행 (scout_mini, velodyne, d455 camera, genz_icp)
# ROS 2 런치 시스템이 실행할 노드와 다른 런치 파일들을 묶어서 반환하는 표준 함수
# ROS2 launch 명령이 실행될 때 자동으로 호출되어, 실행할 전체 구성을 LaunchDescription 객체로 만들어 전달한다.
def generate_launch_description(): 
    scout_mini_launch=IncludeLaunchDescription(
        # Scout Mini 로봇 기본 구동 런치 (ros2_control, base controller 등)
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('scout_mini_base'),
                'launch',
                'base_launch.py'
            )
        ])
    )

    velodyne_launch=IncludeLaunchDescription(
        # Velodyne VLP-16 LiDAR 드라이버 런치
        # velodyne_driver + velodyne_pointcloud + tf 퍼블리셔 포함
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('velodyne'),
                'launch',
                'velodyne-all-nodes-VLP16-composed-launch.py'
            )
        ])
    )
    d455_launch=IncludeLaunchDescription(
          # Intel RealSense D455 카메라 런치 (컬러+깊이 스트리밍, TF 포함)
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('d455_custom_launch'),
                'launch',
                'custom.launch.py'
            )
        ])
    )

    # genz_icp_odometry_launch = IncludeLaunchDescription(
    #     # GenZ-ICP 기반 LiDAR 오도메트리 노드 런치
    #     # (scan matching 기반 pose estimation)
    #     PythonLaunchDescriptionSource([
    #         '/home/marin/marine/src/genz_icp/ros/launch/odometry.launch.py'
    #     ])
    # )

    # rko_lio = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([
    #         '/home/marin/marine/src/rko_lio/ros/launch/odometry.launch.py'
    #     ])
    # )



    #RVIZ2 시각화 툴 
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output = 'screen',
    )

    # LaunchDescription 반환
    # → 실행 시 위의 모든 노드/런치파일 병렬 실행
    return LaunchDescription(
        [
            scout_mini_launch,
            velodyne_launch,
            d455_launch,
            #genz_icp_odometry_launch,
            #rko_lio,
            rviz
        ]
    )
    