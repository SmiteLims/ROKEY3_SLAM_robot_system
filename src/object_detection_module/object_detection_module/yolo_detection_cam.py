# ROS2 Python 라이브러리 import
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time
from std_msgs.msg import Int32

# ROS2 이미지 메시지 타입과 OpenCV-ROS 브리지
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# YOLO 모델 로드 (ultralytics 라이브러리 사용)
from ultralytics import YOLO
import cv2
import numpy as np


# YOLO 기반 객체 감지 노드 클래스 정의
class YOLODetectionNode(Node): # 사각지대 CCTV 노드
    def __init__(self):
        super().__init__('yolo_detection_node') # 노드 이름 설정

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ROI 이미지를 퍼블리시할 퍼블리셔 생성
        # self.publisher_ = self.create_publisher(Image, '/roi_image', 10) # QoS 다시 설정 예정
        self.publisher_ = self.create_publisher(Image, '/roi_image', qos_profile)
        
        # 상태 퍼블리셔 (객체가 사라지면 1 발행)
        self.status_publisher_ = self.create_publisher(Int32, '/status', 10)
        self.last_detect_time = time.time()  # 초기화
        self.status_sent = False  # 중복 발행 방지

        # OpenCV와 ROS 메시지 간 변환을 위한 브리지 객체
        self.bridge = CvBridge()

        # 학습된 YOLO 모델 로드 (.pt 파일 경로)
        self.model = YOLO('/home/rokey/rokey_ws/src/rokey_pjt/rokey_pjt/config/best.pt')

        # USB 또는 기타 카메라 장치 열기 (장치 번호 2번)
        self.cap = cv2.VideoCapture(2)

        # 감지할 타겟 클래스 ID (예: 0번이 자동차)
        self.target_class_id = 0

        # 클래스 ID에 대한 이름 매핑
        self.class_names = {0: 'car'}

        # 카메라 오픈 실패 시 에러 출력 후 노드 종료
        if not self.cap.isOpened():
            self.get_logger().error("Camera open failed")
            rclpy.shutdown()
            return

        self.get_logger().info("YOLO Detection Node Started")

        # 주기적으로 실행될 타이머 콜백 (10Hz → 0.1초마다 실행)
        self.timer = self.create_timer(0.1, self.timer_callback)

        # 결과를 보여줄 창 설정 (수동 사이즈 조정 가능하게)
        self.window_name = "YOLO Detection"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    # 타이머 콜백 함수: 주기적으로 프레임을 받아와 객체 탐지 수행
    def timer_callback(self):
        # 프레임 캡처 시도
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Frame capture failed")
            return

        # YOLO 모델을 사용하여 객체 감지 수행
        results = self.model(frame, stream=False)
        
        object_detected = False  # 매 타이머마다 초기화

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])  # 감지된 객체 클래스 ID
                if cls == self.target_class_id:  # 관심 있는 클래스만 처리
                    object_detected = True  # 감지되면 True
                    self.last_detect_time = time.time()  # 시간 업데이트
                    self.status_sent = False  # 다시 감지되었으니 초기화

                    # 경계 상자 좌표 가져오기
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # ROI(관심 영역) 잘라내기
                    roi = frame[y1:y2, x1:x2]

                    if roi.size == 0:
                        self.get_logger().warn("Empty ROI, skipping...")
                        continue

                    # ROI를 2배 확대 (보여주기 용도)
                    roi_enlarged = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

                    # ROS 이미지 메시지로 변환 후 퍼블리시
                    roi_msg = self.bridge.cv2_to_imgmsg(roi_enlarged, encoding='bgr8')
                    self.publisher_.publish(roi_msg)

                    # 디버깅 용도: 화면에 경계 상자와 클래스 이름 표시
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = self.class_names.get(cls, f"class_{cls}")
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 3초 이상 감지 안 되면 상태 메시지 발행
        if not object_detected and not self.status_sent:
            if time.time() - self.last_detect_time > 3.0:
                msg = Int32()
                msg.data = 1
                self.status_publisher_.publish(msg)
                self.get_logger().info("No object detected for 3 seconds. Published 1 to /status")
                self.status_sent = True  # 중복 발행 방지

        # 창 사이즈를 현재 프레임에 맞게 조정
        h, w = frame.shape[:2]
        cv2.resizeWindow(self.window_name, w, h)

        # 화면에 프레임 출력
        cv2.imshow(self.window_name, frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            self.get_logger().info("Shutting down node...")
            rclpy.shutdown()

# 메인 함수: ROS2 노드 초기화 및 실행
def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectionNode()
    rclpy.spin(node)  # 노드가 종료될 때까지 루프 실행

# 스크립트 직접 실행 시 main 호출
if __name__ == '__main__':
    main()