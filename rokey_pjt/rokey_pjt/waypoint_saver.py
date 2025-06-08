#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from ament_index_python.packages import get_package_share_directory
import os
import math
import yaml


class WaypointSaver(Node):
    def __init__(self):
        super().__init__('waypoint_saver')

        # 자동 증가 인덱스
        self.current_index = 0

        # YAML 경로 설정
        pkg_share = get_package_share_directory('rokey_pjt')
        config_dir = os.path.join(pkg_share, 'config')
        os.makedirs(config_dir, exist_ok=True)
        self.yaml_path = os.path.join(config_dir, 'new_waypoints.yaml')

        # 기존 파일에서 마지막 index 이어받기
        self.data = {'waypoints': []}
        if os.path.isfile(self.yaml_path):
            try:
                with open(self.yaml_path, 'r', encoding='utf-8') as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict) and 'waypoints' in loaded:
                        self.data = loaded
                        existing_ids = [wp.get('id', '') for wp in self.data['waypoints']]
                        # wp_숫자 형식만 필터링
                        wp_nums = [int(wp.split('_')[1]) for wp in existing_ids if wp.startswith('wp_') and wp.split('_')[1].isdigit()]
                        if wp_nums:
                            self.current_index = max(wp_nums) + 1
            except Exception as e:
                self.get_logger().warn(f'기존 YAML 파일 읽기 실패 ({e}), 새로 생성합니다.')

        # 토픽 구독
        self.sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/robot1/initialpose',
            self.pose_callback,
            10
        )
        self.get_logger().info(f'WaypointSaver 연속 저장 모드 시작 (초기 index={self.current_index})')

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        # 쿼터니언 → yaw(deg)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = qw * qw + qx * qx - qy * qy - qz * qz
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        yaw_deg = math.degrees(yaw_rad)

        # 자동 waypoint ID
        waypoint_id = f'wp_{self.current_index}'
        self.current_index += 1

        new_wp = {
            'id': waypoint_id,
            'x': float(f"{x:.6f}"),
            'y': float(f"{y:.6f}"),
            'yaw': float(f"{yaw_deg:.4f}"),
            'neighbors': []
        }

        self.data['waypoints'].append(new_wp)

        # 저장
        try:
            with open(self.yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            self.get_logger().info(f'[{waypoint_id}] 저장 완료 → x: {x:.2f}, y: {y:.2f}, yaw: {yaw_deg:.2f}°')
        except Exception as e:
            self.get_logger().error(f'YAML 저장 실패: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = WaypointSaver()
    if rclpy.ok():
        rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
