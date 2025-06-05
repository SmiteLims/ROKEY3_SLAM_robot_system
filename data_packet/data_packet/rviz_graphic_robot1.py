import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PointStamped
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge    
import numpy as np
import matplotlib.pyplot as plt
import cv2
import threading

class MapOverlayDashboard(Node):
    def __init__(self):
        super().__init__('map_overlay_dashboard')
        self.map_msg = None
        self.plan_msg = None
        self.costmap_msg = None
        self.robot_pose = None

        self.detected_objects = []  # 탐지된 객체 저장
        self.lock = threading.Lock()

        self.bridge = CvBridge()   

        self.create_subscription(OccupancyGrid, '/robot1/map', self.cb_map, 10)
        self.create_subscription(Path, '/robot1/plan', self.cb_plan, 10)
        self.create_subscription(OccupancyGrid, '/robot1/global_costmap/costmap', self.cb_costmap, 10)
        self.create_subscription(PoseStamped, '/robot1/amcl_pose', self.cb_pose, 10)  
        self.create_subscription(PointStamped, '/robot1/detected_object_map_point', self.cb_detected_obj, 10)

        self.publisher = self.create_publisher(Image, '/robot1/map_dashboard', 10)  

        self.declare_parameter("show_costmap", True)
        self.timer = self.create_timer(1/10, self.render)

    def cb_map(self, msg):
        with self.lock:
            self.map_msg = msg

    def cb_plan(self, msg):
        with self.lock:
            self.plan_msg = msg

    def cb_costmap(self, msg):
        with self.lock:
            self.costmap_msg = msg

    def cb_pose(self, msg):
        with self.lock:
            self.robot_pose = msg.pose

    def cb_detected_obj(self, msg):
        # msg: PointStamped (frame_id='map')
        with self.lock:
            self.detected_objects.append((msg.point.x, msg.point.y, msg.point.z, msg.header.stamp))
            if len(self.detected_objects) > 50:
                self.detected_objects = self.detected_objects[-50:]

    def costmap_to_color(self, costarr):
        cmap = plt.get_cmap('jet')  # 'jet', 'hot', etc
        normed = costarr.astype(np.float32)
        normed[costarr < 0] = 0      # unknown은 0으로
        normed = normed / 100.0      # 0~1 사이로 정규화
        normed[normed > 1.0] = 1.0   # 100 이상은 1.0
        colors = (cmap(normed)[:, :, :3] * 255).astype(np.uint8)  # RGB 반환
        return colors

    def render(self):
        # 데이터 복사(스레드 안전하게)
        with self.lock:
            map_msg = self.map_msg
            plan_msg = self.plan_msg
            costmap_msg = self.costmap_msg
            robot_pose = self.robot_pose
            detected_copy = list(self.detected_objects)  # shallow copy

        if map_msg is None:
            return

        w, h = map_msg.info.width, map_msg.info.height
        arr = np.array(map_msg.data, dtype=np.int8).reshape((h, w))
        map_img = np.full((h, w, 3), 180, np.uint8)
        map_img[arr == 0] = [255, 255, 255]
        map_img[arr == 100] = [0, 0, 0]

        # Costmap overlay
        if self.get_parameter("show_costmap").get_parameter_value().bool_value and costmap_msg is not None:
            costarr = np.array(costmap_msg.data, dtype=np.int8).reshape(
                (costmap_msg.info.height, costmap_msg.info.width))
            cost_rgb = self.costmap_to_color(costarr)
            cost_rgb = cv2.resize(cost_rgb, (w, h), interpolation=cv2.INTER_NEAREST)

            mask = costarr > 0
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

            alpha = 0.45
            map_img[mask] = cv2.addWeighted(map_img[mask], 1 - alpha, cost_rgb[mask], alpha, 0)

        # Path overlay
        if plan_msg is not None and len(plan_msg.poses) > 1:
            ox = map_msg.info.origin.position.x
            oy = map_msg.info.origin.position.y
            res = map_msg.info.resolution
            for i in range(len(plan_msg.poses)-1):
                p1 = plan_msg.poses[i].pose.position
                p2 = plan_msg.poses[i+1].pose.position
                x1 = int((p1.x - ox) / res)
                y1 = int((p1.y - oy) / res)
                x2 = int((p2.x - ox) / res)
                y2 = int((p2.y - oy) / res)
                y1 = h - y1 - 1
                y2 = h - y2 - 1
                cv2.line(map_img, (x1, y1), (x2, y2), (220, 100, 20), 3)

        # Robot 위치 표시
        rx, ry = None, None
        if robot_pose is not None:
            x = robot_pose.position.x
            y = robot_pose.position.y
            ox = map_msg.info.origin.position.x
            oy = map_msg.info.origin.position.y
            res = map_msg.info.resolution
            rx = int((x - ox) / res)
            ry = int((y - oy) / res)
            ry = h - ry - 1

        # 맵 이미지 2배 확대
        map_img = cv2.resize(map_img, (w*2, h*2), interpolation=cv2.INTER_NEAREST)
        # 확대 후 좌표도 2배로 곱함
        if rx is not None and ry is not None:
            rx2 = rx * 2
            ry2 = ry * 2
            if 0 <= rx2 < w*2 and 0 <= ry2 < h*2:
                cv2.circle(map_img, (rx2, ry2), 4, (0, 0, 255), -1)

        # Object detect 마킹
        if len(detected_copy) > 0:
            ox = map_msg.info.origin.position.x
            oy = map_msg.info.origin.position.y
            res = map_msg.info.resolution
            for (x, y, z, stamp) in detected_copy:
                u = int((x - ox) / res)
                v = int((y - oy) / res)
                v = h - v - 1
                u_draw = u * 2
                v_draw = v * 2
                if 0 <= u_draw < w*2 and 0 <= v_draw < h*2:
                    cv2.circle(map_img, (u_draw, v_draw), 7, (0, 255, 0), -1)  # 초록색

        # cv2.imshow("Robot0 Map Overlay", map_img)  
        # cv2.waitKey(1)

        # === ROS 이미지로 Publish ===
        msg = self.bridge.cv2_to_imgmsg(map_img, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MapOverlayDashboard()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
