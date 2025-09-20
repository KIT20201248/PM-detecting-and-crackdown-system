#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowWaypoints
from rclpy.action import ActionClient
import math

class WaypointSender(Node):
    def __init__(self):
        super().__init__('waypoint_sender')
        self.client = ActionClient(self, FollowWaypoints, 'follow_waypoints')
        self.timer = self.create_timer(3.0, self.send_waypoints)

    def send_waypoints(self):
        self.timer.cancel()
        if not self.client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('FollowWaypoints action server not ready.')
            return

        waypoints = [
            self.create_pose(1.0, 0.0, 0.0),
            self.create_pose(2.0, 0.5, 0.0),
            self.create_pose(3.0, 0.0, math.pi / 2)
        ]

        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = waypoints

        self.get_logger().info('Sending waypoints...')
        self._send_future = self.client.send_goal_async(goal_msg)

    def create_pose(self, x, y, yaw):
        pose = PoseStamped()
        pose.header.frame_id = 'odom'  # 사용자 요청에 따라 map → odom
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        qx, qy, qz, qw = self.quaternion_from_euler(0.0, 0.0, yaw)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def quaternion_from_euler(self, roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr *sp * cy
        return (qx, qy, qz, qw)

def main(args=None):
    rclpy.init(args=args)
    node = WaypointSender()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
