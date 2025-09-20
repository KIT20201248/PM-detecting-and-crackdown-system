import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import os
import datetime

now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"record_{now}.mp4"

class CudaImageSubscriber(Node):
    def __init__(self):
        super().__init__('cuda_image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/d455_camera/color/image_raw',
            self.listener_callback,
            10
        )
        self.bridge = CvBridge()
        self.get_logger().info('🎥 CUDA Image Subscriber Initialized')

        # 녹화 설정
        self.video_writer = None
        self.output_path = os.path.join("/media/marin/4cca4ad9-422b-4ad3-b582-3f9c402dd434/home/omo/videos", filename)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 또는 'avc1', 'H264'
        self.fps = 30.0  # 프레임 속도 기본값
        self.frame_size = None

    def listener_callback(self, msg):
        try:
            # ROS 이미지 → OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # GPU 업로드 및 다운로드 (효과 없이 GPU round-trip만 수행)
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(cv_image)
            result = gpu_image.download()

            # 비디오 녹화기 초기화 (프레임 크기 기준)
            if self.video_writer is None:
                self.frame_size = (result.shape[1], result.shape[0])
                self.video_writer = cv2.VideoWriter(
                    self.output_path, self.fourcc, self.fps, self.frame_size
                )
                self.get_logger().info(f'📁 녹화 시작: {self.output_path}')

            # 영상 녹화
            self.video_writer.write(result)

            # 화면 표시
            cv2.imshow("CUDA Pass-through Image", result)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Failed to process image: {e}')

    def destroy_node(self):
        # 자원 해제
        if self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info(f'✅ 저장 완료: {self.output_path}')
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CudaImageSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 종료 신호 감지됨 (Ctrl+C)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
