#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener, TransformException

class TfSanityCheck(Node):
    """가장 기본적인 TF 변환 기능만 테스트하는 노드"""
    def __init__(self):
        super().__init__('tf_sanity_check_node')
        
        # use_sim_time 파라미터를 선언하고 값을 확인합니다.
        self.declare_parameter('use_sim_time', False)
        if self.get_parameter('use_sim_time').get_parameter_value().bool_value:
            self.get_logger().info("Using simulation time (use_sim_time = True)")
        else:
            self.get_logger().warn("NOT using simulation time (use_sim_time = False). This test will likely fail.")
            
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 복잡한 커스텀 토픽 대신, bag 파일에 확실히 있는 /velodyne_points를 구독합니다.
        self.subscription = self.create_subscription(
            PointCloud2,
            '/velodyne_points',
            self.listener_callback,
            10)
        
        self.get_logger().info('TF Sanity Check Node started. Waiting for /velodyne_points...')
        self.success_reported = False

    def listener_callback(self, msg: PointCloud2):
        # 성공 메시지가 한번만 출력되도록 함
        if self.success_reported:
            return

        source_frame = msg.header.frame_id
        target_frame = 'odom'
        self.get_logger().info(f"Received message. Attempting to transform from '{source_frame}' to '{target_frame}'...")
        
        try:
            # rosbag 재생에 가장 확실한 "가장 최신 시간"으로 변환을 시도
            self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=1.0))
            
            self.get_logger().info(f"=====================================================")
            self.get_logger().info(f"          SUCCESS! TF LOOKUP IS WORKING.             ")
            self.get_logger().info(f"=====================================================")
            self.success_reported = True
            
        except TransformException as ex:
            self.get_logger().error(f"--> FAILED to transform: {ex}")

def main(args=None):
    rclpy.init(args=args)
    node = TfSanityCheck()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()