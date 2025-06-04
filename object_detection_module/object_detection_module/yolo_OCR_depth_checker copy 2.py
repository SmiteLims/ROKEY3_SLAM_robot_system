import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import easyocr
import numpy as np
import cv2
import threading
import os
import sys
import supervision as sv  # ByteTrack 및 Annotators

# ========================
# 설정
# ========================
YOLO_MODEL_PATH = '/home/kim/rokey_ws/runs/detect/car_best.pt'
RGB_TOPIC = '/robot1/oakd/rgb/image_raw'
DEPTH_TOPIC = '/robot1/oakd/stereo/image_raw'
CAMERA_INFO_TOPIC = '/robot1/oakd/stereo/camera_info'
TARGET_CLASS_ID = 0  # 예: car
WINDOW_NAME = "YOLO + OCR + ByteTrack + Depth"
DEPTH_SCALE = 1.17

# ========================
# 클래스 정의
# ========================
class YoloOCRDepthChecker(Node):
    def __init__(self):
        super().__init__('yolo_ocr_depth_checker')
        self.get_logger().info("YOLO + OCR + Depth + ByteTrack 노드 시작")

        # YOLO 모델 로드
        if not os.path.exists(YOLO_MODEL_PATH):
            self.get_logger().error(f"YOLO 모델 경로가 존재하지 않음: {YOLO_MODEL_PATH}")
            sys.exit(1)
        self.model = YOLO(YOLO_MODEL_PATH)
        self.class_names = self.model.names if hasattr(self.model, 'names') else {0: "car"}

        self.reader = easyocr.Reader(['ko'])
        self.bridge = CvBridge()

        self.K = None
        self.rgb_image = None
        self.depth_image = None
        self.lock = threading.Lock()
        self.ocr_done = False

        # ByteTrack 및 Annotators
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30,
            minimum_consecutive_frames=1
        )
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        # ✅ 퍼블리셔
        self.image_publisher_ = self.create_publisher(Image, '/yolo_ocr/image', 10)
        self.plate_publisher_ = self.create_publisher(String, '/yolo_ocr/plate', 10)

        # 서브스크립션
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, 1)
        self.create_subscription(Image, RGB_TOPIC, self.rgb_callback, 1)
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_callback, 1)

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

    def processing_loop(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while rclpy.ok():
            with self.lock:
                if self.rgb_image is None or self.depth_image is None or self.K is None:
                    continue
                rgb = self.rgb_image.copy()
                depth = self.depth_image.copy()

            # YOLO 감지 및 추적
            results = self.model(rgb, verbose=False)[0]
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

                # Depth 정보 추출
                x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0, [depth.shape[1]-1, depth.shape[0]-1, depth.shape[1]-1, depth.shape[0]-1])
                roi_depth = depth[y1:y2, x1:x2]
                roi_filtered = cv2.bilateralFilter(roi_depth.astype(np.float32), 9, 75, 75)
                valid_depths = roi_filtered[(roi_filtered > 100) & (roi_filtered < 5000)]

                if len(valid_depths) == 0:
                    continue

                dist_m = np.median(valid_depths) / 1000.0 if depth.dtype == np.uint16 else float(np.median(valid_depths))
                dist_m *= DEPTH_SCALE
                label_text += f" {dist_m:.2f}m"
                self.get_logger().info(label_text)

                # OCR (최초 1회만 실행)
                if not self.ocr_done:
                    roi = rgb[y1:y2, x1:x2]
                    ocr_result = self.reader.readtext(roi)
                    for text_bbox in ocr_result:
                        text = text_bbox[1].strip()
                        if len(text) >= 3 and any(c.isdigit() for c in text):
                            (ox1, oy1), (ox2, oy2) = map(lambda pt: (int(pt[0]) + x1, int(pt[1]) + y1), [text_bbox[0][0], text_bbox[0][2]])
                            cv2.rectangle(annotated, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
                            cv2.putText(annotated, text, (ox1, oy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            self.get_logger().info(f"[OCR] 인식된 텍스트: {text}")

                            # ✅ 텍스트 퍼블리시
                            plate_msg = String()
                            plate_msg.data = text
                            self.plate_publisher_.publish(plate_msg)

                            self.ocr_done = True
                            break

                # Annotate 결과
                single_detection = sv.Detections(
                    xyxy=np.array([tracked.xyxy[i]]),
                    class_id=np.array([class_id]),
                    tracker_id=np.array([tracker_id]),
                    confidence=np.array([confidence])
                )
                annotated = self.box_annotator.annotate(scene=annotated, detections=single_detection)
                annotated = self.label_annotator.annotate(scene=annotated, detections=single_detection, labels=[label_text])

            # ✅ 결과 이미지 퍼블리시
            ros_image = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            self.image_publisher_.publish(ros_image)

            # OpenCV 디스플레이
            cv2.imshow(WINDOW_NAME, annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# ========================
# 메인 함수
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
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()