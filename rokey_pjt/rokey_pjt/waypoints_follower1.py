#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped, Point, Quaternion
import yaml
import os
import math
import time
import threading
from ament_index_python.packages import get_package_share_directory


def euler_to_quaternion(yaw_rad):
    """
    yaw(rad) → quaternion (roll=0, pitch=0)
    """
    cy = math.cos(yaw_rad * 0.5)
    sy = math.sin(yaw_rad * 0.5)
    return Quaternion(x=0.0, y=0.0, z=sy, w=cy)


def create_pose(x, y, yaw_deg, clock):
    """
    x, y (meter), yaw (degree) → PoseStamped
    """
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = clock.now().to_msg()
    pose.pose.position = Point(x=x, y=y, z=0.0)

    yaw_rad = math.radians(yaw_deg)
    pose.pose.orientation = euler_to_quaternion(yaw_rad)
    return pose


class WaypointsFollower(Node):
    def __init__(self):
        super().__init__('waypoints_follower1')

        # 1) BasicNavigator 인스턴스 생성 + Nav2 준비 대기
        self.navigator = BasicNavigator()
        time.sleep(1.0)
        self.navigator.waitUntilNav2Active()
        self.get_logger().info('[WaypointsFollower1] Nav2 is ready.')

        # 2) YAML에서 waypoint 좌표 로드
        try:
            yaml_path = os.path.join(
                get_package_share_directory('rokey_pjt'),
                'config',
                'waypoints_1.yaml'
            )
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            self.waypoint_dict = {
                wp['id']: {
                    'x': float(wp['x']),
                    'y': float(wp['y']),
                    'yaw': float(wp.get('yaw', 0.0))
                }
                for wp in data.get('waypoints', [])
            }
            self.valid_ids = set(self.waypoint_dict.keys())
            self.get_logger().info(f'[WaypointsFollower1] Loaded {len(self.valid_ids)} waypoints from {yaml_path}')
        except Exception as e:
            self.get_logger().error(f'[WaypointsFollower1] Failed to load waypoints: {e}')
            self.waypoint_dict = {}
            self.valid_ids = set()

        # 3) 상태 플래그
        self.busy = False  # 경로 수행 중인지 여부
        self.nav_thread = None

        # 4) 구독: /robot1/bfs/path
        self.subscription = self.create_subscription(
            String,
            '/robot1/bfs/path',
            self.path_callback,
            10
        )
        self.get_logger().info('[WaypointsFollower1] Subscribed to /robot1/bfs/path')

    def path_callback(self, msg: String):
        """
        "/robot1/bfs/path"으로 들어온 comma-separated waypoint ID 문자열을 처리.
        두 번째부터 끝까지의 waypoints를 PoseStamped 리스트로 만들어
        별도 스레드에서 navigator.goThroughPoses() 실행
        """
        if self.busy:
            self.get_logger().info('[WaypointsFollower1] Already navigating; ignoring new path.')
            return

        raw = msg.data.strip()
        if not raw:
            self.get_logger().warn('[WaypointsFollower1] Received empty path string.')
            return

        waypoint_ids = [wp_id.strip() for wp_id in raw.split(',') if wp_id.strip()]
        self.get_logger().info(f'[WaypointsFollower1] Received path IDs: {waypoint_ids}')

        if len(waypoint_ids) <= 1:
            self.get_logger().warn('[WaypointsFollower1] Path length ≤ 1; no waypoints to follow.')
            return

        next_ids = waypoint_ids[1:]
        invalid = [wid for wid in next_ids if wid not in self.valid_ids]
        if invalid:
            self.get_logger().error(f'[WaypointsFollower1] Invalid waypoint IDs in path: {invalid}')
            return

        # PoseStamped 리스트 생성
        poses = []
        for wid in next_ids:
            coords = self.waypoint_dict[wid]
            pose = create_pose(coords['x'], coords['y'], coords['yaw'], self.navigator.get_clock())
            poses.append(pose)

        if not poses:
            self.get_logger().warn('[WaypointsFollower1] No valid waypoints to navigate.')
            return

        # 비동기 스레드로 네비게이션 실행
        self.busy = True
        self.nav_thread = threading.Thread(target=self._navigate_sequence, args=(poses, next_ids), daemon=True)
        self.nav_thread.start()

    def _navigate_sequence(self, poses, next_ids):
        """
        별도 스레드에서 순차적으로 goThroughPoses 호출 및 결과 처리
        """
        try:
            self.get_logger().info(f'[WaypointsFollower1] Following {len(poses)} waypoints: {next_ids}')
            self.navigator.goThroughPoses(poses)
            result = self.navigator.waitForTaskComplete()

            if result == TaskResult.SUCCEEDED:
                self.get_logger().info('[WaypointsFollower1] All waypoints reached successfully.')
            else:
                self.get_logger().warn(f'[WaypointsFollower1] Waypoints navigation failed (result={result}).')
        except Exception as e:
            self.get_logger().error(f'[WaypointsFollower1] Navigation thread exception: {e}')
        finally:
            self.busy = False

    def destroy_node(self):
        # 노드 종료 시 이동 취소하고 스레드 정리
        try:
            if self.busy:
                self.navigator.cancelAllGoals()
            if self.nav_thread and self.nav_thread.is_alive():
                self.get_logger().info('[WaypointsFollower1] Waiting for navigation thread to finish...')
                self.nav_thread.join(timeout=1.0)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WaypointsFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
