#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime

class VideoRecorderNode(Node):
    def __init__(self):
        super().__init__('video_recorder_node')

        # 저장할 디렉토리
        self.output_dir = '/media/marin/4cca4ad9-422b-4ad3-b582-3f9c402dd434/home/omo/videos/yolo'
        os.makedirs(self.output_dir, exist_ok=True)

        # 파일 이름에 타임스탬프 포함
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_path = os.path.join(self.output_dir, f'yolo_{timestamp}.mp4')

        # 설정값
        self.bridge = CvBridge()
        self.video_writer = None
        self.fps = 30.0  # 프레임 속도
        self.encoding = 'bgr8'
        self.last_frame = None

        # 이미지 구독
        self.subscription = self.create_subscription(
            Image,
            '/camera/d455_camera/color/image_raw',
            self.image_callback,
            10
        )

        self.get_logger().info(f'[✅ 녹화 시작] 저장 위치: {self.output_path}')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, self.encoding)
        except Exception as e:
            self.get_logger().error(f'[❌ 변환 실패] cv_bridge: {e}')
            return

        if self.video_writer is None:
            height, width, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))

        self.video_writer.write(frame)
        self.last_frame = frame

        # 영상 실시간 출력
        cv2.imshow("Recording - Press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info('[🛑 사용자 종료 - q 입력]')
            rclpy.shutdown()

    def destroy_node(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info(f'[💾 저장 완료] {self.output_path}')
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VideoRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('[🛑 사용자 종료 - Ctrl+C]')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

