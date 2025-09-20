#velodyne 라이다 urdf 실행 
import os
import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    urdf_file = os.path.join(
        get_package_share_directory('velodyne_description'),
        'urdf',
        'vlp16.urdf.xacro'
    )

    robot_state_publisher = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': open(urdf_file).read()}]
    )

    return launch.LaunchDescription([robot_state_publisher])

