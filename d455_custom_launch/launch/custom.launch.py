"""
이 런치 파일은 Intel RealSense D455 카메라 드라이버 노드를 실행한다.
- params_file 런치 인자를 통해 사용할 파라미터 YAML 파일 경로를 지정
- 기본 파라미터 파일은 d455_custom_launch 패키지의 params/d455_config.yaml
- 실행 시 realsense2_camera_node를 'd455_camera'라는 이름으로 실행
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    #params_file 런치 인자를 선언하고 기본값으로 d455_config.yaml 파일 경로 설정
    param_dir = LaunchConfiguration(
        "params_file",
        default=os.path.join(
            get_package_share_directory("d455_custom_launch"), "params", "d455_config.yaml"
        ),
    )
    # LaunchDescription에 실행할 노드 목록을 담아 반환
    return LaunchDescription([
        # 런치 인자 선언
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='d455_camera',
            output='screen',
            parameters=[param_dir],
        )
    ])