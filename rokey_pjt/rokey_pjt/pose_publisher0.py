#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import String
from nav2_simple_commander.robot_navigator import BasicNavigator
import yaml
import os
import math
import time
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

amcl_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    depth=10  # 일반적인 기본값
)


def quaternion_from_yaw(yaw_deg):
    """yaw(도 단위) → quaternion(z, w) 반환"""
    yaw_rad = math.radians(yaw_deg)
    qz = math.sin(yaw_rad / 2.0)
    qw = math.cos(yaw_rad / 2.0)
    return qz, qw


def create_pose(x, y, yaw_deg, navigator):
    """x, y, yaw(도 단위) → PoseStamped 생성"""
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x = x
    pose.pose.position.y = y

    qz, qw = quaternion_from_yaw(yaw_deg)
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw

    return pose


class WaypointInitialPoseSetter(Node):
    def __init__(self):
        super().__init__('pose_publisher0')

        # Nav2 Navigator 생성
        self.navigator = BasicNavigator()

        # 기본으로 사용할 시작 waypoint ID
        self.target_id = 'init_pose'

        # waypoints YAML 로딩
        try:
            yaml_path = os.path.join(
                get_package_share_directory('rokey_pjt'),
                'config',
                'waypoints_0.yaml'
            )
            with open(yaml_path, 'r') as f:
                self.waypoints = yaml.safe_load(f)['waypoints']
                self.get_logger().info(f'Waypoints loaded from: {yaml_path}')
        except Exception as e:
            self.get_logger().error(f"Failed to load waypoint config: {e}")
            self.waypoints = []

        # publisher: 상태 ID를 publish할 토픽 (절대 경로)
        self.state_pub = self.create_publisher(
            String,
            '/robot0/bfs/state_pose',
            10
        )

        # threshold (meter)
        self.threshold = 0.8  

        # 1) 초기 pose 설정
        self.set_initial_pose()

        # 2) AMCL 피드백 구독 - 절대 경로 사용
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/robot0/amcl_pose',
            self.pose_callback,
            amcl_qos
        )

    def set_initial_pose(self):
        # waypoint ID로 초기 pose 찾기
        target = next((wp for wp in self.waypoints if wp['id'] == self.target_id), None)
        if not target:
            self.get_logger().error(f"Waypoint ID '{self.target_id}' not found.")
            return

        pose = create_pose(target['x'], target['y'], target['yaw'], self.navigator)
        self.navigator.setInitialPose(pose)
        self.get_logger().info(f"초기 위치 설정 중... → ({target['x']:.2f}, {target['y']:.2f}, yaw: {target['yaw']} deg)")

        # AMCL & TF 안정화 대기
        time.sleep(10.0)
        self.navigator.waitUntilNav2Active()
        self.get_logger().info("Nav2 활성화 완료. 초기 pose 설정 끝.")

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y
        self.get_logger().info(f'[DEBUG] 현재 위치: ({current_x:.2f}, {current_y:.2f})')

        closest_id = None
        min_dist = float('inf')

        # 각 waypoint와의 거리 계산
        for wp in self.waypoints:
            dx = current_x - wp['x']
            dy = current_y - wp['y']
            dist = math.hypot(dx, dy)
            self.get_logger().info(f"[DEBUG] ↪ WP '{wp['id']}': 위치=({wp['x']:.2f}, {wp['y']:.2f}), 거리={dist:.2f}m")

            if dist < min_dist:
                min_dist = dist
                closest_id = wp['id']

        self.get_logger().info(f'[DEBUG] 가장 가까운 waypoint: {closest_id} (거리: {min_dist:.2f} m)')

        # threshold 이하일 때마다 매번 publish
        if closest_id and min_dist <= self.threshold:
            msg_out = String()
            msg_out.data = closest_id
            self.state_pub.publish(msg_out)
            self.get_logger().info(f'[WaypointState] → 퍼블리시: {closest_id} (거리: {min_dist:.2f} m)')


def main(args=None):
    rclpy.init(args=args)
    node = WaypointInitialPoseSetter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
