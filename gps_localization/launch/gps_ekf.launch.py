from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    params_file = LaunchConfiguration('params_file')

    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value='/home/marin/marine/src/gps_localization/params/ekf_gps.yaml',
        description='Path to YAML file with robot_localization + navsat_transform parameters'
    )

    ekf_odom_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node_odom',
        output='screen',
        parameters=[params_file],
        arguments=['--ros-args', '--log-level', 'info'],
        remappings=[("odometry/filtered", "odom/local_ekf")],
    )

    ekf_map_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node_map',
        output='screen',
        parameters=[params_file],
        arguments=['--ros-args', '--log-level', 'info'],
        remappings=[("odometry/filtered", "odom/global_ekf")],
    )

    navsat_node = Node(
        package='robot_localization',
        executable='navsat_transform_node',
        name='navsat_transform_node',
        output='screen',
        parameters=[params_file],
        arguments=['--ros-args', '--log-level', 'info'],
        remappings=[
            ("/imu", "/mavros/imu/data"),
            ("gps/fix", "/mavros/global_position/global"),
            ("odometry/gps", "odometry/gps"),
            ("gps/filtered", "gps/filtered"),
            ("odometry/filtered", "odom/global_ekf")
        ]
    )

    return LaunchDescription([
        declare_params_file,
        ekf_odom_node,
        ekf_map_node,
        navsat_node
    ])
