#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String
import threading
import time


class MissionPlanner(Node):
    def __init__(self):
        super().__init__('controller0')

        # 상태 변수
        self.goal_sequence = ['p2_2', 'p2_3', 'p2_4']
        self.return_goal = 'p1_0'
        self.current_index = 0
        self.current_pose = None
        self.is_mission_started = False
        self.lock = threading.Lock()

        # Publisher
        self.goal_pub = self.create_publisher(String, '/robot0/bfs/goal_pose', 10)

        # Subscribers
        self.create_subscription(Int32, '/robot0/system_alert', self.start_mission_callback, 10)
        self.create_subscription(Int32, '/robot0/db/detect', self.detect_callback, 10)
        self.create_subscription(String, '/robot0/bfs/state_pose', self.state_pose_callback, 10)

        self.get_logger().info('[MissionPlanner] Node ready. Waiting for system alert.')

    def start_mission_callback(self, msg: Int32):
        if msg.data == 1 and not self.is_mission_started:
            self.get_logger().info('[MissionPlanner] Mission start signal received.')
            self.is_mission_started = True
            self.current_index = 0
            threading.Thread(target=self.run_mission, daemon=True).start()

    def detect_callback(self, msg: Int32):
        if msg.data == 1:
            self.get_logger().warn('[MissionPlanner] Detected alert! Returning to p1_0.')
            with self.lock:
                self.is_mission_started = False
                self.current_index = 0
            self.publish_goal(self.return_goal)

            # 상태 포즈가 return_goal(p1_0)이 될 때까지 대기
            def wait_until_arrival():
                while rclpy.ok():
                    if self.current_pose == self.return_goal:
                        self.get_logger().info('[MissionPlanner] Arrived at p1_0 after detection. Shutting down.')
                        rclpy.shutdown()
                        break
                    time.sleep(0.5)

        threading.Thread(target=wait_until_arrival, daemon=True).start()


    def state_pose_callback(self, msg: String):
        self.current_pose = msg.data.strip()

    def run_mission(self):
        while True:
            with self.lock:
                if not self.is_mission_started or self.current_index >= len(self.goal_sequence):
                    self.get_logger().info('[MissionPlanner] Mission complete or interrupted.')
                    break

                next_goal = self.goal_sequence[self.current_index]

                if self.current_pose == next_goal:
                    self.get_logger().info(f'[MissionPlanner] Already at {next_goal}. Skipping.')
                    self.current_index += 1
                    continue

                self.publish_goal(next_goal)
                self.current_index += 1

            time.sleep(1.0)

    def publish_goal(self, goal_id: str):
        self.goal_pub.publish(String(data=goal_id))
        self.get_logger().info(f'[MissionPlanner] Published goal_pose: {goal_id}')


def main(args=None):
    rclpy.init(args=args)
    node = MissionPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()