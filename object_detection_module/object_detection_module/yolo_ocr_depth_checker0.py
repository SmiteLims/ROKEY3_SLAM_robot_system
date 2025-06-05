import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import easyocr
import numpy as np
import cv2
import threading
import os
import sys
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Int32
from vehicle_info.msg import VehicleInfo
import re
# ========================
# 설정
# ========================
YOLO_MODEL_PATH = '/home/kim/rokey_ws/runs/detect/plate_fix_best.pt'
RGB_TOPIC = '/robot0/oakd/rgb/image_fast/compressed'
DEPTH_TOPIC = '/robot0/oakd/stereo/image_raw'  # ✅ Raw depth 토픽으로 변경
CAMERA_INFO_TOPIC = '/robot0/oakd/stereo/camera_info'
TARGET_CLASS_ID = 0
WINDOW_NAME = "YOLO + OCR Distance View"
# ========================

class YoloOCRDepthChecker(Node):
    def __init__(self):
        super().__init__('yolo_ocr_depth_checker')
        self.get_logger().info("YOLO + OCR + Depth 노드 시작")

        if not os.path.exists(YOLO_MODEL_PATH):
            self.get_logger().error(f"YOLO 모델 경로 없음: {YOLO_MODEL_PATH}")
            sys.exit(1)
        self.model = YOLO(YOLO_MODEL_PATH)
        self.class_names = self.model.names if hasattr(self.model, 'names') else []

        self.cnt=0
        self.reader = easyocr.Reader(['en'])
        self.bridge = CvBridge()
        self.K = None
        self.rgb_image = None
        self.depth_image = None
        self.depth_encoding = None
        self.near_event = False
        self.distance_threshold_m = 0.7
        self.lock = threading.Lock()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(CompressedImage, RGB_TOPIC, self.rgb_callback, qos)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, 1)
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_callback, 10)  # ✅ 변경된 부분
        
        self.distance_publisher = self.create_publisher(Int32, '/robot0/event', qos)        # 거리용
        self.vehicle_info_publisher = self.create_publisher(VehicleInfo, 'vehicle_info', 10)  # 번호판용

        threading.Thread(target=self.processing_loop, daemon=True).start()

    def camera_info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info("카메라 내부 파라미터 수신 완료")

    def rgb_callback(self, msg):
        with self.lock:
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.rgb_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def depth_callback(self, msg):  # ✅ Image 타입으로 받음
        with self.lock:
            try:
                depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
                if depth.dtype == np.uint16:
                    self.depth_encoding = 'uint16'
                elif depth.dtype == np.float32:
                    self.depth_encoding = 'float32'
                else:
                    self.get_logger().warn(f"지원되지 않는 depth dtype: {depth.dtype}")
                    return
                self.depth_image = cv2.medianBlur(depth, 5)
            except Exception as e:
                self.get_logger().warn(f"Depth 디코딩 실패: {e}")

    def processing_loop(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while rclpy.ok():
            with self.lock:
                if self.rgb_image is None or self.depth_image is None or self.K is None:
                    continue
                rgb = self.rgb_image.copy()
                depth = self.depth_image.copy()

            detected_regions = []
            results = self.model(rgb, stream=False, verbose=False)[0]

            for box in results.boxes:
                cls = int(box.cls[0])
                if cls != TARGET_CLASS_ID:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                u, v = (x1 + x2) // 2, (y1 + y2) // 2

                if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
                    continue

                val = depth[v, u]
                distance_m = val / 1000.0 if depth.dtype == np.uint16 else float(val)
                if distance_m == 0.0 or np.isnan(distance_m):
                    continue

                label = self.class_names[cls] if cls < len(self.class_names) else f'class_{cls}'
                self.get_logger().info(f"{label} at ({u},{v}) → {distance_m:.2f}m")

                # ✅ /event 퍼블리시
                near_event = Int32()
                if distance_m < self.distance_threshold_m: # and not self.near_event:
                    near_event.data = 1
                    self.distance_publisher.publish(near_event)
                    self.near_event = True
                else :
                    ############################
                    if self.cnt == 2:
                        near_event.data = 0
                        self.distance_publisher.publish(near_event)
                        self.cnt = 0
                    else:
                        self.cnt+=1
                    #######################
                # 시각화
                cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(rgb, (u, v), 4, (0, 0, 255), -1)
                cv2.putText(rgb, f"{distance_m:.2f}m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                detected_regions.append((x1, y1, x2, y2))

            for x1, y1, x2, y2 in detected_regions:
                roi = rgb[y1:y2, x1:x2]
                ocr_results = self.reader.readtext(roi)

                for i in ocr_results:
                    text = i[1].strip()
                    digits_only = ''.join(filter(str.isdigit, text))
                    if not re.fullmatch(r'\d{4}', digits_only):
                        continue

                    # ✅ /vehicle_info 퍼블리시
                    try:
                        plate_number = int(digits_only)
                        # plate_msg = Int32()
                        plate_msg = VehicleInfo()
                        plate_msg.id = plate_number
                        plate_msg.location = "Site 1"
                        self.vehicle_info_publisher.publish(plate_msg)
                        self.get_logger().info(f"OCR 번호판 인식: {plate_number}")
                    except ValueError:
                        continue  # 숫자 추출 실패

                    ox1 = int(i[0][0][0]) + x1
                    oy1 = int(i[0][0][1]) + y1
                    ox2 = int(i[0][2][0]) + x1
                    oy2 = int(i[0][2][1]) + y1

                    cv2.rectangle(rgb, (ox1, oy1), (ox2, oy2), (255, 0, 255), 2)
                    cv2.putText(rgb, text, (ox1, oy1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow(WINDOW_NAME, rgb)
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
