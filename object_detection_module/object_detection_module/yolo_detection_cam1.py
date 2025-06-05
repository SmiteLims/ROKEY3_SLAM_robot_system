import cv2
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage  # 압축 이미지 메시지
from std_msgs.msg import Int32  # 상태 메시지
from cv_bridge import CvBridge
from threading import Thread, Lock
import time

class YOLOCompressedCameraNode(Node):
    def __init__(self):
        super().__init__('yolo_compressed_camera_node')

        # YOLO 모델 로드
        self.model = YOLO('/home/kim/rokey_ws/runs/detect/car_best.pt')

        # 카메라 설정
        self.cap = cv2.VideoCapture(4)
        if not self.cap.isOpened():
            self.get_logger().error("USB 카메라 열기 실패.")
            rclpy.shutdown()    
            return

        # 퍼블리셔: 압축된 이미지 + 상태
        self.image_publisher_ = self.create_publisher(CompressedImage, '/robot1/cam/image/compressed', 10)
        self.status_publisher_ = self.create_publisher(Int32, '/robot1/status', 10)

        # CvBridge
        self.bridge = CvBridge()

        # 스레드 락
        self.lock = Lock()
        self.running = True

        # 감지 여부 관리 (업그레이드)
        self.detected_once = False
        self.last_detect_time = time.time()
        self.current_status = 0  # 0: 감지 중, 1: 감지 안 됨

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

            # YOLO 추론
            results = self.model(frame, stream=False, conf=0.6, verbose=False)

            object_detected = False

            annotated = frame.copy()
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    confidence = round(float(box.conf[0]), 2)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"car: {confidence:.2f}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    object_detected = True

            current_time = time.time()

            # 차량 감지되었을 때
            if object_detected:
                if self.current_status == 1:
                    # 이전에 1 보냈으면 복구 (0) 발행
                    msg = Int32()
                    msg.data = 0
                    self.status_publisher_.publish(msg)
                    self.get_logger().info("Car detected again. Published 0 to /robot1/status")
                self.detected_once = True
                self.last_detect_time = current_time
                self.current_status = 0

            # 차량이 한 번이라도 감지된 이후, 현재 감지 안 됨
            if self.detected_once and not object_detected:
                if (current_time - self.last_detect_time > 3.0) and self.current_status == 0:
                    msg = Int32()
                    msg.data = 1
                    self.status_publisher_.publish(msg)
                    self.get_logger().info("No detection for 3 seconds. Published 1 to /robot1/status")
                    self.current_status = 1

            # 프레임 저장
            with self.lock:
                self.annotated_frame = annotated

            # ROS2 압축 이미지 퍼블리시
            ros_compressed_image = self.bridge.cv2_to_compressed_imgmsg(annotated, dst_format='jpeg')
            self.image_publisher_.publish(ros_compressed_image)

    def run_loop(self):
        WINDOW_NAME = "YOLO Detection"
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while rclpy.ok() and self.running:
            with self.lock:
                frame = getattr(self, 'annotated_frame', None)

            if frame is not None:
                cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.yolo_thread.join()

def main(args=None):
    rclpy.init(args=args)
    node = YOLOCompressedCameraNode()
    try:
        rclpy.spin(node)  # 노드가 종료될 때까지 spin
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()