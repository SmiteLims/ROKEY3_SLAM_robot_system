# ROS2 관련 패키지 임포트
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import String, Int32
from vehicle_info.msg import VehicleInfo
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge

# 객체 감지 및 OCR 관련 패키지
from ultralytics import YOLO
import easyocr
import numpy as np
import cv2
import threading
import os
import sys
import supervision as sv

# QoS 설정 관련
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import time

# ========================
# 설정값
# ========================
YOLO_MODEL_PATH = '/home/kim/rokey_ws/runs/detect/plate_best.pt'
RGB_TOPIC = '/robot0/oakd/rgb/image_raw'
DEPTH_TOPIC = '/robot0/oakd/stereo/image_raw'
CAMERA_INFO_TOPIC = '/robot0/oakd/stereo/camera_info'
TARGET_CLASS_ID = 0
DEPTH_SCALE = 1.0

# ========================
# 메인 클래스 정의
# ========================
class YoloOCRDepthChecker(Node):
    def __init__(self):
        super().__init__('yolo_ocr_depth_checker')
        self.get_logger().info("YOLO + OCR + Depth + ByteTrack 노드 시작")

        if not os.path.exists(YOLO_MODEL_PATH):
            self.get_logger().error(f"YOLO 모델 경로가 존재하지 않음: {YOLO_MODEL_PATH}")
            sys.exit(1)

        self.model = YOLO(YOLO_MODEL_PATH)
        self.class_names = self.model.names if hasattr(self.model, 'names') else {0: "car"}
        self.reader = easyocr.Reader(['en'])
        self.bridge = CvBridge()

        self.K = None
        self.rgb_image = None
        self.depth_image = None
        self.lock = threading.Lock()
        self.ocr_done = False

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        qos_profile_reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        qos_profile2 = QoSProfile(depth=10, durability=DurabilityPolicy.VOLATILE, reliability=ReliabilityPolicy.RELIABLE)
        qos_profile_vehicle = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30,
            minimum_consecutive_frames=1
        )
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        # compressed 이미지 퍼블리셔
        self.image_publisher_ = self.create_publisher(CompressedImage, '/robot0/yolo_ocr/image/compressed', qos_profile)
        self.vehicle_info_publisher_ = self.create_publisher(VehicleInfo, 'vehicle_info', qos_profile_vehicle)
        self.distance_publisher_ = self.create_publisher(Int32, '/robot0/yolo_ocr/near', qos_profile)

        self.create_subscription(PoseWithCovarianceStamped, '/robot0/amcl_pose', self.amcl_pose_callback, qos_profile2)
        self.position_data = PoseWithCovarianceStamped()

        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, 1)
        self.create_subscription(Image, RGB_TOPIC, self.rgb_callback, 1)
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_callback, 1)

        self.distance_threshold_m = 0.8
        self.near_event_sent = False

        threading.Thread(target=self.processing_loop, daemon=True).start()

    def camera_info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info("카메라 내부 파라미터 수신 완료")

    def rgb_callback(self, msg):
        with self.lock:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def depth_callback(self, msg):
        with self.lock:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def amcl_pose_callback(self, msg):
        self.position_data = msg

    def processing_loop(self):
        while rclpy.ok():
            with self.lock:
                if self.rgb_image is None or self.depth_image is None or self.K is None:
                    continue
                rgb = self.rgb_image.copy()
                depth = self.depth_image.copy()

            rgb_h, rgb_w = rgb.shape[:2]
            depth_h, depth_w = depth.shape[:2]

            try:
                results = self.model(rgb, verbose=False)[0]
            except Exception as e:
                self.get_logger().warn(f"YOLO 추론 실패: {e}")
                continue

            detections = sv.Detections.from_ultralytics(results)
            detections = detections[detections.class_id == TARGET_CLASS_ID]
            tracked = self.tracker.update_with_detections(detections)

            annotated = rgb.copy()

            for i in range(len(tracked)):
                x1, y1, x2, y2 = map(int, tracked.xyxy[i])
                class_id = tracked.class_id[i]
                tracker_id = tracked.tracker_id[i]
                confidence = tracked.confidence[i]
                label_text = f"{self.class_names[class_id]} #{tracker_id}"

                cx_rgb = (x1 + x2) // 2
                cy_rgb = (y1 + y2) // 2
                cx_depth = int(cx_rgb * (depth_w / rgb_w))
                cy_depth = int(cy_rgb * (depth_h / rgb_h))

                roi = depth[max(0, cy_depth - 2):min(depth_h, cy_depth + 3),
                            max(0, cx_depth - 2):min(depth_w, cx_depth + 3)]
                valid_roi = roi[(roi > 0) & (roi < 5000)]

                if valid_roi.size == 0:
                    continue

                center_depth = np.median(valid_roi)

                dist_m = float(center_depth) / 1000.0 if depth.dtype == np.uint16 else float(center_depth)
                dist_m *= DEPTH_SCALE
                label_text += f" {dist_m:.2f}m"

                if dist_m < self.distance_threshold_m and not self.near_event_sent:
                    near_msg = Int32()
                    near_msg.data = 1
                    self.distance_publisher_.publish(near_msg)
                    self.near_event_sent = True
                elif dist_m >= self.distance_threshold_m:
                    self.near_event_sent = False

                if not self.ocr_done:
                    roi = rgb[y1:y2, x1:x2]
                    try:
                        ocr_result = self.reader.readtext(roi)
                    except Exception as e:
                        self.get_logger().warn(f"OCR 처리 중 예외 발생: {e}")
                        continue

                    for text_bbox in ocr_result:
                        text = text_bbox[1].strip()
                        if len(text) == 4 and text.isdigit():
                            self.get_logger().info(f"[OCR] 인식된 텍스트: {text}")
                            vehicle_info_msg = VehicleInfo()
                            vehicle_info_msg.id = int(text)
                            vehicle_info_msg.location = "Site 0"
                            self.vehicle_info_publisher_.publish(vehicle_info_msg)
                            self.get_logger().info(f"[VehicleInfo] 차량 번호: {text} 퍼블리시 완료")
                            self.ocr_done = True
                            break

                    if self.ocr_done:
                        break

                single_detection = sv.Detections(
                    xyxy=np.array([tracked.xyxy[i]]),
                    class_id=np.array([class_id]),
                    tracker_id=np.array([tracker_id]),
                    confidence=np.array([confidence])
                )
                annotated = self.box_annotator.annotate(scene=annotated, detections=single_detection)
                annotated = self.label_annotator.annotate(scene=annotated, detections=single_detection, labels=[label_text])

            # JPEG 압축 후 CompressedImage로 퍼블리시
            compressed_msg = CompressedImage()
            compressed_msg.header.stamp = self.get_clock().now().to_msg()
            compressed_msg.header.frame_id = "camera_frame"
            compressed_msg.format = "jpeg"
            _, buffer = cv2.imencode('.jpg', annotated)
            compressed_msg.data = buffer.tobytes()
            self.image_publisher_.publish(compressed_msg)

            time.sleep(0.01)

# ========================
# 메인 실행 함수
# ========================
def main():
    rclpy.init()
    node = YoloOCRDepthChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()