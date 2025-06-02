import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO
import easyocr
import numpy as np
import cv2
import threading
import os
import sys

# ========================
# 설정
# ========================
YOLO_MODEL_PATH = '/home/kim/rokey_ws/runs/detect/car_best.pt'
RGB_TOPIC = '/robot0/oakd/rgb/image_raw'
DEPTH_TOPIC = '/robot0/oakd/stereo/image_raw'
CAMERA_INFO_TOPIC = '/robot0/oakd/stereo/camera_info'
TARGET_CLASS_ID = 0  # 예: car
WINDOW_NAME = "YOLO + OCR Distance View"
DEPTH_SCALE = 1.17  # 🆕 보정 계수
# ========================

class YoloOCRDepthChecker(Node): # 터틀봇 차량인식이랑 번호판 인식
    def __init__(self):
        super().__init__('yolo_ocr_depth_checker')
        self.get_logger().info("YOLO + OCR + Depth 노드 시작")

        if not os.path.exists(YOLO_MODEL_PATH):
            self.get_logger().error(f"YOLO 모델 경로가 존재하지 않음: {YOLO_MODEL_PATH}")
            sys.exit(1)
        self.model = YOLO(YOLO_MODEL_PATH)
        self.class_names = {0: "car"}

        self.reader = easyocr.Reader(['ko'])

        self.bridge = CvBridge()
        self.K = None
        self.rgb_image = None
        self.depth_image = None
        self.lock = threading.Lock()
        self.ocr_done = False

        self.create_subscription(CameraInfo, CAMERA_INFO_TOPrgb_callbackIC, self.camera_info_callback, 1)
        self.create_subscription(Image, RGB_TOPIC, self., 1)
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_callback, 1)

        threading.Thread(target=self.processing_loop, daemon=True).start()

    def camera_info_callback(self, msg): # 카메라 제대로 작동하는지 체크용
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info("카메라 내부 파라미터 수신 완료")

    def rgb_callback(self, msg): # RGB 카메라 이미지 수신
        with self.lock:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def depth_callback(self, msg): # Depth 이미지 수신
        with self.lock:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def processing_loop(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while rclpy.ok():
            with self.lock:
                if self.rgb_image is None or self.depth_image is None or self.K is None:
                    continue

                # 
                rgb = self.rgb_image.copy()
                depth = self.depth_image.copy()

            detected_regions = []
            results = self.model(rgb, stream=True)
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls != TARGET_CLASS_ID:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0, [depth.shape[1]-1, depth.shape[0]-1, depth.shape[1]-1, depth.shape[0]-1])
                    roi_depth = depth[y1:y2, x1:x2]

                    # 🆕 Bilateral filter 적용
                    roi_depth_filtered = cv2.bilateralFilter(roi_depth.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)

                    # 🆕 유효 depth 값 필터링
                    valid_depths = roi_depth_filtered[(roi_depth_filtered > 100) & (roi_depth_filtered < 5000)]
                    if len(valid_depths) == 0:
                        continue

                    val = np.median(valid_depths)
                    distance_m = val / 1000.0 if depth.dtype == np.uint16 else float(val)
                    distance_m *= DEPTH_SCALE  # 보정

                    label = self.class_names[cls] if cls in self.class_names else f'class_{cls}'
                    self.get_logger().info(f"{label} at [{x1},{y1},{x2},{y2}] → {distance_m:.2f}m")

                    # 시각화
                    cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(rgb, f"{label} {distance_m:.2f}m", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    detected_regions.append((x1, y1, x2, y2))

            # OCR: YOLO 감지된 영역만 대상으로, OCR은 한 번만 수행
            if not self.ocr_done and detected_regions:
                for x1, y1, x2, y2 in detected_regions:
                    roi = rgb[y1:y2, x1:x2]
                    ocr_results = self.reader.readtext(roi)

                    for i in ocr_results:
                        text = i[1].strip()
                        if len(text) < 3 or not any(c.isdigit() for c in text):
                            continue

                        ox1 = int(i[0][0][0]) + x1
                        oy1 = int(i[0][0][1]) + y1
                        ox2 = int(i[0][2][0]) + x1
                        oy2 = int(i[0][2][1]) + y1

                        cv2.rectangle(rgb, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
                        cv2.putText(rgb, text, (ox1, oy1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        self.get_logger().info(f"[OCR] 인식된 텍스트: {text}")
                        self.ocr_done = True
                        break

                    if self.ocr_done:
                        break

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

