import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import easyocr
import cv2
import re
from collections import Counter

class OCRProcessingNode(Node): # OCR Preprocessing # 필요하지 않다 지금
    def __init__(self):
        super().__init__('ocr_processing_node')
        self.subscription = self.create_subscription(Image, '/roi_image', self.listener_callback, 10)
        self.bridge = CvBridge()
        self.reader = easyocr.Reader(['ko'])
        self.get_logger().info("OCR Processing Node Started")

    def listener_callback(self, msg):
        roi = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 초기 OCR 수행 (테스트로 1회만 진행해서 번호판 유무 확인)
        ocr_results = self.reader.readtext(roi)

        if ocr_results:
            largest_box = max(ocr_results, key=lambda x: (x[0][2][0] - x[0][0][0]) * (x[0][2][1] - x[0][0][1]))
            text = largest_box[1].strip()
            cleaned_text = self.clean_text(text)

            if self.is_valid_plate(cleaned_text):
                print(f"Initial OCR Detected Plate: {cleaned_text}")

                # 10번 OCR 반복
                ocr_texts = []
                for _ in range(10):
                    repeat_results = self.reader.readtext(roi)
                    if repeat_results:
                        largest_box_repeat = max(repeat_results, key=lambda x: (x[0][2][0] - x[0][0][0]) * (x[0][2][1] - x[0][0][1]))
                        text_repeat = largest_box_repeat[1].strip()
                        cleaned_text_repeat = self.clean_text(text_repeat)
                        if self.is_valid_plate(cleaned_text_repeat):
                            ocr_texts.append(cleaned_text_repeat)

                if ocr_texts:
                    # 다수결 (Voting)
                    most_common_plate, count = Counter(ocr_texts).most_common(1)[0]
                    print(f"Voted Plate Result: {most_common_plate} (voted {count} times)")

                    # 시각화
                    (x1, y1) = (int(largest_box[0][0][0]), int(largest_box[0][0][1]))
                    (x2, y2) = (int(largest_box[0][2][0]), int(largest_box[0][2][1]))
                    cv2.rectangle(roi, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(roi, most_common_plate, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("ROI Enlarged + OCR", roi)
        cv2.waitKey(1)

    def clean_text(self, text):
        text = re.sub(r'[\(\)\s\-]', '', text)
        return text

    def is_valid_plate(self, text):
        if len(text) != 7:
            return False
        pattern = r'^\d{2}[가-힣]\d{4}$'
        if re.match(pattern, text):
            return True
        return False

def main(args=None):
    rclpy.init(args=args)
    node = OCRProcessingNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()