import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np

class YOLODetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_detection_node')
        self.publisher_ = self.create_publisher(Image, '/roi_image', 10)
        self.bridge = CvBridge()
        self.model = YOLO('/home/rokey/rokey_ws/src/rokey_pjt/rokey_pjt/config/best.pt')
        self.cap = cv2.VideoCapture(2)
        self.target_class_id = 0
        self.class_names = {0: 'car'}

        if not self.cap.isOpened():
            self.get_logger().error("Camera open failed")
            rclpy.shutdown()
            return

        self.get_logger().info("YOLO Detection Node Started")
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz

        # 창 이름 설정
        self.window_name = "YOLO Detection"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)  # 윈도우를 수동 사이즈 가능하게 설정

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Frame capture failed")
            return

        results = self.model(frame, stream=False)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == self.target_class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi = frame[y1:y2, x1:x2]

                    # ROI 2배 확대
                    roi_enlarged = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

                    # ROI Publish
                    roi_msg = self.bridge.cv2_to_imgmsg(roi_enlarged, encoding='bgr8')
                    self.publisher_.publish(roi_msg)

                    # Optional: Show detection frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = self.class_names.get(cls, f"class_{cls}")
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 프레임 사이즈에 맞춰 창 크기 조정
        h, w = frame.shape[:2]
        cv2.resizeWindow(self.window_name, w, h)

        cv2.imshow(self.window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            self.get_logger().info("Shutting down node...")
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectionNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()