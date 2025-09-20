# Copyright 2019 Open Source Robotics Foundation, Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
- Velodyne VLP-16 LiDAR 전체 구동 런치 (드라이버 + 포인트클라우드 변환 + LaserScan 변환)
- velodyne_driver: LiDAR 원시 패킷 수신 및 디코딩
- velodyne_pointcloud (Transform): 캘리브레이션 적용 및 3D 포인트클라우드 생성
- velodyne_laserscan: 포인트클라우드를 LaserScan 토픽으로 변환
- velodyne_description: URDF(xacro) 기반 로봇모델 로드 및 TF 브로드캐스트
- ComposableNodeContainer를 사용하여 모든 LiDAR 관련 노드를 하나의 프로세스에서 실행
"""

import os
import yaml

import ament_index_python.packages
from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import launch_ros.actions
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
     # ── Velodyne LiDAR 드라이버 설정 로드 ─────────────────────────────
    driver_share_dir = ament_index_python.packages.get_package_share_directory('velodyne_driver')
    driver_params_file = os.path.join(driver_share_dir, 'config', 'VLP16-velodyne_driver_node-params.yaml')
    with open(driver_params_file, 'r') as f:
        driver_params = yaml.safe_load(f)['velodyne_driver_node']['ros__parameters']

    # ── 포인트클라우드 변환 노드 설정 로드 (캘리브레이션 포함) ───────────
    convert_share_dir = ament_index_python.packages.get_package_share_directory('velodyne_pointcloud')
    convert_params_file = os.path.join(convert_share_dir, 'config', 'VLP16-velodyne_transform_node-params.yaml')
    with open(convert_params_file, 'r') as f:
        convert_params = yaml.safe_load(f)['velodyne_transform_node']['ros__parameters']
    convert_params['calibration'] = os.path.join(convert_share_dir, 'params', 'VLP16db.yaml')
    
     # ── LaserScan 변환 노드 설정 로드 ──────────────────────────────────
    laserscan_share_dir = ament_index_python.packages.get_package_share_directory('velodyne_laserscan')
    laserscan_params_file = os.path.join(laserscan_share_dir, 'config', 'default-velodyne_laserscan_node-params.yaml')
    with open(laserscan_params_file, 'r') as f:
        laserscan_params = yaml.safe_load(f)['velodyne_laserscan_node']['ros__parameters']

    # ── URDF(xacro) 파일 경로 지정 ─────────────────────────────────────
    xacro_file = PathJoinSubstitution([
    FindPackageShare("velodyne_description"),
    "urdf",
    "vlp16.urdf.xacro"
    ])
    
    # ── URDF(xacro) → robot_description 변환 ──────────────────────────
    velodyne_description_content = Command([
    FindExecutable(name="xacro"),
    " ",  # 공백 추가 (필수)
    xacro_file
    ])
    
     # ── TF 브로드캐스트용 robot_state_publisher 실행 ────────────────────
    robot_state_publisher_node = launch_ros.actions.Node(
    package="robot_state_publisher",
    executable="robot_state_publisher",
    output="screen",
    parameters=[{"robot_description": velodyne_description_content}],
    remappings=[("/robot_description", "/velodyne_description")]
    )

    # ── Velodyne LiDAR 관련 노드를 하나의 컨테이너에서 실행 ─────────────
    container = ComposableNodeContainer(
            name='velodyne_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                 # LiDAR 드라이버 (원시 데이터 수신)
                ComposableNode(
                    package='velodyne_driver',
                    plugin='velodyne_driver::VelodyneDriver',
                    name='velodyne_driver_node',
                    parameters=[driver_params]),
                # 포인트클라우드 변환 (좌표계 변환 + 보정)
                ComposableNode(
                    package='velodyne_pointcloud',
                    plugin='velodyne_pointcloud::Transform',
                    name='velodyne_transform_node',
                    parameters=[convert_params]),
                # 포인트클라우드 → LaserScan 변환
                ComposableNode(
                    package='velodyne_laserscan',
                    plugin='velodyne_laserscan::VelodyneLaserScan',
                    name='velodyne_laserscan_node',
                    parameters=[laserscan_params]),
            ],
            output='both',
    )
     # ── LaunchDescription 반환 (컨테이너 + URDF 브로드캐스터) ──────────
    return LaunchDescription([
        container,
        robot_state_publisher_node
        ])
