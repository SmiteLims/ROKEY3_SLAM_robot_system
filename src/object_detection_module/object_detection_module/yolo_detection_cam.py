import cv2
from ultralytics import YOLO
import easyocr
import time
import psutil
import pandas as pd
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32  # 상태 토픽
from threading import Thread, Lock
from collections import deque

class YOLOOCRCameraNode(Node):
    def __init__(self):
        super().__init__('yolo_ocr_camera_node')

        # YOLO 모델 설정
        self.model_path = '/home/rokey/rokey_ws/src/rokey_pjt/rokey_pjt/config/best.pt'
        self.model = YOLO(self.model_path)

        # 결과 저장 폴더
        self.output_dir = './output'
        if os.path.exists(self.output_dir):
            os.system(f'rm -rf {self.output_dir}')
        os.makedirs(self.output_dir)

        # 타겟 클래스
        self.target_class_id = 0
        self.class_names = {0: "car"}

        # OCR Reader
        self.reader = easyocr.Reader(['ko'])

        # 카메라 설정
        self.cap = cv2.VideoCapture(2)  # USB 카메라
        if not self.cap.isOpened():
            self.get_logger().error("USB 카메라 열기 실패.")
            rclpy.shutdown()
            return

        # 퍼포먼스 측정
        self.warmup_time = 5
        self.measure_frames = 100
        self.measure = False
        self.frame_skip = 3  # 매 프레임마다 추론
        self.frame_count = 0
        self.measuring = False
        self.start_time = time.time()
        self.measure_data = []
        self.csv_output = []
        self.confidences = []

        # 멀티스레드용
        self.lock = Lock()
        self.annotated_frame = deque(maxlen=1)
        self.running = True

        # OCR 1회만 수행
        self.ocr_once = True
        self.ocr_done = False

        # 감지 상태 관리
        self.last_detect_time = time.time()
        self.status_sent = False

        # ROS2 퍼블리셔
        self.status_publisher_ = self.create_publisher(Int32, '/robot1/status', 10)

        # self.get_logger().info("YOLO OCR Camera Node Initialized.")

        # YOLO 추론 스레드 시작
        self.yolo_thread = Thread(target=self.yolo_loop)
        self.yolo_thread.start()

        # 메인 루프 (GUI)
        self.run_loop()

    def yolo_loop(self):
        while self.running and rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("프레임 수신 실패")
                continue

            now = time.time()
            if not self.measuring and now - self.start_time >= self.warmup_time:
                # self.get_logger().info("[측정 시작]")
                self.measuring = True

            if self.frame_count % self.frame_skip == 0:
                frame_start = time.perf_counter()

                results = self.model(frame, stream=False, conf=0.6,verbose=False)

                frame_time = time.perf_counter() - frame_start
                cpu_percent = psutil.cpu_percent(interval=None)

                new_detections = []
                object_detected = False
                object_count = 0

                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        if cls != self.target_class_id:
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = round(float(box.conf[0]), 2)
                        label = self.class_names.get(cls, f"class_{cls}")
                        self.confidences.append(confidence)

                        new_detections.append((x1, y1, x2, y2, confidence, label))
                        self.csv_output.append([x1, y1, x2, y2, confidence, label])
                        object_detected = True
                        object_count += 1

                # 3초 이상 감지 안 되면 1 발행
                if not object_detected and not self.status_sent:
                    if time.time() - self.last_detect_time > 3.0:
                        msg = Int32()
                        msg.data = 1
                        self.status_publisher_.publish(msg)
                        self.get_logger().info("No object detected for 3 seconds. Published 1 to /status")
                        self.status_sent = True
                        msg.data = 0

                # 감지되면 마지막 감지 시간 업데이트
                if object_detected:
                    self.last_detect_time = time.time()
                    self.status_sent = False

                # 결과 업데이트
                with self.lock:
                    self.last_detections = new_detections

                # OCR 1회 수행
                if not self.ocr_done and self.ocr_once and new_detections:
                    for x1, y1, x2, y2, _, _ in new_detections:
                        roi = frame[y1:y2, x1:x2]
                        ocr_results = self.reader.readtext(roi)
                        for i in ocr_results:
                            text = i[1].strip()
                            if len(text) < 3 or not any(c.isdigit() for c in text):
                                continue
                            ox1 = int(i[0][0][0]) + x1
                            oy1 = int(i[0][0][1]) + y1
                            ox2 = int(i[0][2][0]) + x1
                            oy2 = int(i[0][2][1]) + y1
                            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
                            cv2.putText(frame, text, (ox1, oy1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            # self.get_logger().info(f"[OCR] 인식된 텍스트: {text}")
                            self.ocr_done = True

                if object_count > 0:
                    filename = f'frame_{int(time.time())}.jpg'
                    cv2.imwrite(os.path.join(self.output_dir, filename), frame)

                if self.measuring:
                    self.measure_data.append({
                        'frame': self.frame_count,
                        'frame_time': frame_time,
                        'cpu_percent': cpu_percent
                    })

                    if self.measure and len(self.measure_data) >= self.measure_frames:
                        # self.get_logger().info("[측정 완료]")
                        self.save_results()
                        self.running = False
                        break

            # 프레임 업데이트
            with self.lock:
                annotated = frame.copy()
                for x1, y1, x2, y2, confidence, label in getattr(self, 'last_detections', []):
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"{label}: {confidence}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                self.annotated_frame.clear()
                self.annotated_frame.append(annotated)

            self.frame_count += 1

    def run_loop(self):
        WINDOW_NAME = "YOLO + OCR"
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while rclpy.ok():
            with self.lock:
                if len(self.annotated_frame) == 0:
                    continue
                frame = self.annotated_frame[-1]

            if frame is not None:
                cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.yolo_thread.join()
        # self.get_logger().info("카메라 종료 및 노드 종료")

    def save_results(self):
        df = pd.DataFrame(self.measure_data)
        df['fps'] = 1 / df['frame_time']
        df.to_csv(os.path.join(self.output_dir, 'performance.csv'), index=False)

        avg_fps = df['fps'].mean()
        avg_frame_time = df['frame_time'].mean()
        avg_cpu_percent = df['cpu_percent'].mean()

        # self.get_logger().info("===== 성능 측정 결과 (YOLO 프레임 기준) =====")
        # self.get_logger().info(f"평균 FPS: {avg_fps:.2f}")
        # self.get_logger().info(f"평균 프레임 시간: {avg_frame_time:.4f}초")
        # self.get_logger().info(f"평균 CPU 사용률: {avg_cpu_percent:.2f}%")

def main(args=None):
    rclpy.init(args=args)
    node = YOLOOCRCameraNode()
    try:
        rclpy.spin(node)  # Node가 종료될 때까지 spin
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()