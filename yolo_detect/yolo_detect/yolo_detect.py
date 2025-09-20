import os
import cv2
import numpy as np
import datetime
from geopy import Point
from geopy.distance import geodesic

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, NavSatFix
from std_msgs.msg import Float64, Int8
from cv_bridge import CvBridge
from ultralytics import YOLO

from .sort import Sort  # sort.py가 같은 패키지 내부에 있어야 함

# ─────────────────────────────────────────────────────────────────────
# [녹화 파일 이름 설정]
# - now: 현재 시각(YYYYMMDD_HHMMSS)
# - filename: 녹화 mp4 파일명
# ─────────────────────────────────────────────────────────────────────
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"record_{now}.mp4"

# ─────────────────────────────────────────────────────────────────────
# [투영 변환(원근→BEV) 설정 파라미터]
# - SRC_POINTS: 원본(카메라) 영상의 기준 4점 (픽셀 좌표, 1280x720 기준으로 추정)
# - SQUARE_SIZE: 변환 결과(BEV) 정사각형 한 변의 길이(픽셀)
# - DST_POINTS: BEV 목표 좌표계 상의 4점 (0~SQUARE_SIZE 범위)
# - M: cv2.getPerspectiveTransform으로 구한 3x3 투영행렬
#   ※ 이 행렬은 이후 픽셀 좌표를 BEV 평면으로 사상할 때 사용
# ─────────────────────────────────────────────────────────────────────
SRC_POINTS = np.float32([
    [3, 556],
    [1279, 566],
    [843, 419],
    [421, 419],
])
SQUARE_SIZE = 720
DST_POINTS = np.float32([
    [0, SQUARE_SIZE],
    [SQUARE_SIZE, SQUARE_SIZE],
    [SQUARE_SIZE, 0],
    [0, 0],
])
M = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')

        # ───────────────────────────────────────────────────────────────
        # [인도 판정 구독]
        # - /scooter_on_sidewalk(Int8): 세그멘테이션 결과로 '인도 위(1)/아님(0)' 신호 수신
        # - self.sidewalk_result: 최근 판정 결과 보관 (0/1)
        #
        # ★ 요청에 따라 인도 위 여부 이용 로직 전면 비활성화
        # self.sidewalk_result = 0
        # self.sidewalk_sub = self.create_subscription(
        #     Int8,
        #     '/scooter_on_sidewalk',
        #     self.sidewalk_callback,
        #     10
        # )
        # ───────────────────────────────────────────────────────────────

        # ───────────────────────────────────────────────────────────────
        # [카메라 이미지 구독]
        # - /realsense/D555_408222301500_Color(Image): D555 컬러 프레임
        # - 콜백: image_callback
        # - 큐: 10
        # ───────────────────────────────────────────────────────────────
        self.subscription = self.create_subscription(
            Image,
            '/realsense/D555_408222301500_Color',
            self.image_callback,
            10)

        # ───────────────────────────────────────────────────────────────
        # [브리지 및 장치/모델 설정]
        # - CvBridge: ROS Image ↔ OpenCV 변환
        # - self.device: YOLO 추론 장치 지정(ultralytics: 0→GPU:0, 'cpu'도 가능)
        # - self.model: 모델 경로
        # - self.tracker(SORT):
        #     max_age=60, min_hits=3, iou_threshold=0.20
        # - self.detected_coords: (현재는 사용 X) 확정 좌표 보관
        # - self.gps_history: 트랙별 좌표 이력(MAX_HISTORY 제한)
        # ───────────────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.device = 0
        self.model = YOLO('/home/marin/marine/src/yolo_detect/models/640v11.pt')
        self.tracker = Sort(max_age=60, min_hits=3, iou_threshold=0.20)
        self.detected_coords = {}
        self.gps_history = {}

        # ───────────────────────────────────────────────────────────────
        # [퍼블리셔]
        # ★ 요청에 따라 yolo_segmentation으로의 BEV 픽셀/확정 좌표 송신 비활성화
        # self.coord_pub = self.create_publisher(NavSatFix, '/scooter_location', 10)
        # self.pixel_coord_pub = self.create_publisher(NavSatFix, '/scooter_pixel_bev', 10)
        # ───────────────────────────────────────────────────────────────

        # ───────────────────────────────────────────────────────────────
        # [GPS/헤딩 구독]
        # ───────────────────────────────────────────────────────────────
        self.latest_coords = None
        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/mavros/global_position/global',
            self.gps_callback,
            qos_profile_sensor_data)

        self.compass_heading = 0.0
        self.heading_sub = self.create_subscription(
            Float64,
            '/mavros/global_position/compass_hdg',
            self.heading_callback,
            qos_profile_sensor_data)

        # ───────────────────────────────────────────────────────────────
        # [영상 기록 설정]
        # ───────────────────────────────────────────────────────────────
        self.output_path = os.path.join(
            "/media/marin/4cca4ad9-422b-4ad3-b582-3f9c402dd434/home/omo/videos/yolo", filename)
        self.video_writer = None
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = 30.0
        self.frame_size = None

        # ───────────────────────────────────────────────────────────────
        # [로그 파일 초기화]
        # ───────────────────────────────────────────────────────────────
        self.log_file_path = os.path.join("/home/marin/marine/src/yolo_detect", "scooter_location.txt")
        with open(self.log_file_path, 'w') as f:
            f.write("count: 0\n")

    def gps_callback(self, msg: NavSatFix):
        lat, lon = msg.latitude, msg.longitude
        if abs(lat) < 0.0001 or abs(lon) < 0.0001:
            self.latest_coords = None
            return
        self.latest_coords = (lat, lon)

    def heading_callback(self, msg: Float64):
        self.compass_heading = msg.data

    def estimate_scooter_gps(self, lat, lon, heading, x1, y1, x2, y2,
                             image_width=640, hfov_deg=90.0):
        # image_width 기본값을 640 기준으로 변경
        if lat is None or lon is None:
            return None

        width, height = abs(x2 - x1), abs(y2 - y1)
        area = width * height
        if area == 0:
            return None

        distance = 615 / (area ** 0.5)
        bbox_center_x = (x1 + x2) / 2
        image_cx = image_width / 2
        deg_per_pixel = hfov_deg / image_width
        relative_angle = (bbox_center_x - image_cx) * deg_per_pixel
        bearing = (heading + relative_angle) % 360
        origin = Point(lat, lon)
        destination = geodesic(meters=distance).destination(origin, bearing)
        return [destination.latitude, destination.longitude]

    def update_log_file(self):
        lines = [f"count: {len(self.detected_coords)}\n"]
        for tid, coord in self.detected_coords.items():
            if coord is None:
                lines.append(f"id: {tid} | 위도: N/A | 경도: N/A\n")
            else:
                lat, lon = coord
                lines.append(f"id: {lat:.6f} | 위도: {lat:.6f} | 경도: {lon:.6f}\n")
            if tid in self.gps_history:
                lines.append("  └ 이동 기록:\n")
                for i, (lat, lon) in enumerate(self.gps_history[tid]):
                    lines.append(f"     [{i+1:02}] lat: {lat:.6f}, lon: {lon:.6f}\n")
        with open(self.log_file_path, 'w') as f:
            f.writelines(lines)

    # def sidewalk_callback(self, msg: Int8):
    #     # 세그멘테이션 결과(인도 여부) 최신값 저장 (1: 인도 위)
    #     self.sidewalk_result = msg.data

    def image_callback(self, msg):
        MAX_HISTORY = 15

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        resized_frame = cv2.resize(frame, (640, 640))  # 640×640 기준

        results = self.model.predict(
            resized_frame, imgsz=640, conf=0.70, device=self.device, verbose=False
        )

        detections = [[*box.xyxy[0].tolist(), float(box.conf)] for r in results for box in r.boxes
                      if self.model.names[int(box.cls)] == "scooter"]

        dets_np = np.array(detections) if detections else np.empty((0, 5))
        tracks = self.tracker.update(dets_np)

        for track in tracks:
            x1, y1, x2, y2, track_id = track.astype(int)
            bbox_center_x = (x1 + x2) / 2
            # 640 기준에서의 중앙 분기(대략 좌/우 접지점 선택)
            selected_x, selected_y = (x2, y2) if bbox_center_x <= 320 else (x1, y2)

            # 리사이즈(640×640) → 원 해상도(1280×720)로 스케일 복원
            scale_x, scale_y = 1280 / 640, 720 / 640
            x_scaled, y_scaled = selected_x * scale_x, selected_y * scale_y

            # 원근→BEV 투영 (한 점 변환)
            pt = np.array([[[x_scaled, y_scaled]]], dtype=np.float32)
            bev_pt = cv2.perspectiveTransform(pt, M)[0][0]
            u, v = int(bev_pt[0]), int(bev_pt[1])

            # ★ 요청에 따라 yolo_segmentation으로 BEV 픽셀 좌표 송신 비활성화
            # pix_msg = NavSatFix()
            # pix_msg.header.stamp = self.get_clock().now().to_msg()
            # pix_msg.header.frame_id = 'map'
            # pix_msg.latitude = float(v)
            # pix_msg.longitude = float(u)
            # pix_msg.altitude = 0.0
            # pix_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
            # self.pixel_coord_pub.publish(pix_msg)

            coord = self.estimate_scooter_gps(*self.latest_coords, self.compass_heading, x1, y1, x2, y2) \
                    if self.latest_coords is not None else None

            if coord is None:
                continue

            if track_id not in self.gps_history:
                self.gps_history[track_id] = []
            self.gps_history[track_id].append(coord)
            if len(self.gps_history[track_id]) > MAX_HISTORY:
                self.gps_history[track_id] = self.gps_history[track_id][-MAX_HISTORY:]

            # ★ 요청에 따라 '인도 위(=1)일 때만 확정/카운트' 로직 전면 비활성화
            # if track_id not in self.detected_coords and self.sidewalk_result == 1:
            #     self.detected_coords[track_id] = coord
            #     self.update_log_file()
            #
            #     gps_msg = NavSatFix()
            #     gps_msg.header.stamp = self.get_clock().now().to_msg()
            #     gps_msg.header.frame_id = 'base_link'
            #     gps_msg.latitude = coord[0]
            #     gps_msg.longitude = coord[1]
            #     gps_msg.altitude = 0.0
            #     gps_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
            #     self.coord_pub.publish(gps_msg)

            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized_frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(resized_frame, f"Scooter count: {len(self.detected_coords)}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if self.video_writer is None:
            self.frame_size = (resized_frame.shape[1], resized_frame.shape[0])
            self.video_writer = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, self.frame_size)
            self.get_logger().info(f'Started recording to: {self.output_path}')
        self.video_writer.write(resized_frame)

        cv2.imshow("YOLO_detect", resized_frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.video_writer is not None:
            node.video_writer.release()
            node.get_logger().info('Video saved.')
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
