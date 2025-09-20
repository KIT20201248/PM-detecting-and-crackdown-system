from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    

    # 2) gps_localization의 기존 노드 + heading_tools 런치 포함
    return LaunchDescription([
        Node(
            package='gps_localization',
            executable='gps_localization',   # 너희 패키지의 실행파일명
            name='gps_localization',
            output='screen'
        ),

        # include: heading_tools (imu_yaw_enu, compass_to_enu 두 노드가 이 안에서 실행됨
    ])
