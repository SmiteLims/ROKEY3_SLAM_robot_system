#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory
import yaml
import os
from collections import deque


class BFSAndValidatorNode(Node):
    def __init__(self):
        super().__init__('bfs_path_finder0')

        # ─── YAML 로딩 ─────────────────────────────
        try:
            yaml_path = os.path.join(
                get_package_share_directory('rokey_pjt'),
                'config',
                'waypoints_0.yaml'
            )
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            # 그래프 생성
            self.graph = {}
            self.waypoint_dict = {}
            for wp in data['waypoints']:
                wp_id = wp['id']
                self.graph[wp_id] = wp.get('neighbors', [])
                self.waypoint_dict[wp_id] = {
                    'x': float(wp['x']),
                    'y': float(wp['y']),
                    'yaw': float(wp.get('yaw', 0.0))
                }

            # 양방향 연결 보장
            for node, neighbors in self.graph.items():
                for nbr in neighbors:
                    self.graph.setdefault(nbr, [])
                    if node not in self.graph[nbr]:
                        self.graph[nbr].append(node)

            self.valid_ids = set(self.waypoint_dict.keys())
            self.get_logger().info(f"[Init] Loaded {len(self.valid_ids)} waypoints from {yaml_path}")

        except Exception as e:
            self.get_logger().error(f"[Init] Failed to load waypoints: {e}")
            self.graph = {}
            self.waypoint_dict = {}
            self.valid_ids = set()

        # ─── 상태 변수 ──────────────────────────────
        self.current_pose_id = None
        self.goal_pose_id = None

        # ─── Subscribers ────────────────────────────
        self.create_subscription(String, '/robot0/bfs/state_pose', self.state_callback, 10)
        self.create_subscription(String, '/robot0/bfs/goal_pose', self.goal_callback, 10)
        self.create_subscription(String, '/robot0/bfs/path', self.path_callback, 10)

        # ─── Publishers ─────────────────────────────
        self.path_pub = self.create_publisher(String, '/robot0/bfs/path', 10)
        self.pose_pub = self.create_publisher(String, '/robot0/bfs/next_pose', 10)

        self.get_logger().info("[Init] BFS and PathValidator combined node ready.")

    # ────────────────────────────────────────────────
    def state_callback(self, msg):
        self.current_pose_id = msg.data
        self.get_logger().info(f"[State] Current pose set to: {self.current_pose_id}")
        self.try_compute_path()

    def goal_callback(self, msg):
        self.goal_pose_id = msg.data
        self.get_logger().info(f"[Goal] Goal pose set to: {self.goal_pose_id}")
        self.try_compute_path()

    def try_compute_path(self):
        if self.current_pose_id and self.goal_pose_id:
            if self.current_pose_id == self.goal_pose_id:
                self.get_logger().warn("[BFS] Current == Goal. Publishing empty path.")
                self.path_pub.publish(String(data=""))
                return

            path = self.bfs(self.current_pose_id, self.goal_pose_id)
            if path:
                path_str = ','.join(path)
                self.path_pub.publish(String(data=path_str))
                self.get_logger().info(f"[BFS] Published path: {path_str}")
            else:
                self.path_pub.publish(String(data=""))
                self.get_logger().warn(f"[BFS] No path found from {self.current_pose_id} to {self.goal_pose_id}")

    def bfs(self, start_id, goal_id):
        visited = set([start_id])
        queue = deque([(start_id, [start_id])])

        while queue:
            current, path = queue.popleft()
            if current == goal_id:
                return path
            for neighbor in self.graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    # ────────────────────────────────────────────────
    def path_callback(self, msg: String):
        """
        path string 예: "wp1,wp2,wp3"
        → 두 번째 유효한 waypoint를 찾아 (x, y, yaw) 형식으로 publish
        """
        raw = msg.data.strip()
        if not raw:
            self.get_logger().warn('[Validator] Received empty path string.')
            self.publish_empty()
            return

        incoming_ids = [wid.strip() for wid in raw.split(',') if wid.strip()]
        self.get_logger().info(f'[Validator] Received path: {incoming_ids}')

        valid_ids = [wid for wid in incoming_ids if wid in self.valid_ids]
        if len(valid_ids) < 2:
            self.get_logger().warn('[Validator] Less than 2 valid waypoints. Publishing empty.')
            self.publish_empty()
            return

        wp = self.waypoint_dict[valid_ids[1]]
        pose_str = f"[{wp['x']}, {wp['y']}, {wp['yaw']}]"
        self.pose_pub.publish(String(data=pose_str))
        self.get_logger().info(f"[Validator] Published 2nd waypoint pose: {pose_str}")

    def publish_empty(self):
        self.pose_pub.publish(String(data="[]"))
        self.get_logger().info('[Validator] Published empty pose.')


def main(args=None):
    rclpy.init(args=args)
    node = BFSAndValidatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
