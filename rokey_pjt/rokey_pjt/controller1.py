#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String
from irobot_create_msgs.msg import AudioNoteVector, AudioNote
from builtin_interfaces.msg import Duration
import threading
import time 

class MissionPlanner(Node):
    def __init__(self):
        super().__init__('controller0')

        # 상태 변수
        self.goal_sequence = ['p2_2', 'p2_3', 'p2_4']
        self.return_goal = 'p1_0'
        self.current_index = 0
        self.current_goal = None
        self.current_pose = None
        self.is_mission_started = False
        self.is_returning = False
        self.lock = threading.Lock()
        self.timer = None

        # Publishers
        self.goal_pub = self.create_publisher(String, '/robot1/bfs/goal_pose', 10)
        self.audio_pub = self.create_publisher(AudioNoteVector, '/robot1/cmd_audio', 10)

        # Subscribers
        self.create_subscription(Int32, '/robot1/system_alert', self.start_mission_callback, 10)
        self.create_subscription(Int32, '/robot1/db/detect', self.detect_callback, 10)
        self.create_subscription(String, '/robot1/bfs/state_pose', self.state_pose_callback, 10)

        self.get_logger().info('[MissionPlanner] Node ready. Waiting for system alert.')

    def start_mission_callback(self, msg: Int32):
        if msg.data == 1 and not self.is_mission_started:
            self.get_logger().info('[MissionPlanner] Mission start signal received.')
            self.is_mission_started = True
            self.current_index = 0
            self.is_returning = False
            self.publish_next_goal()

    def detect_callback(self, msg: Int32):
        if msg.data == 1:
            self.get_logger().warn('[MissionPlanner] Detected alert! Returning to p1_0.')
            self.send_beep()  # 삐뽀삐보 소리 재생


            with self.lock:
                self.is_mission_started = False
                self.is_returning = True
                self.current_goal = self.return_goal
                self.start_goal_timer(self.return_goal)

    def state_pose_callback(self, msg: String):
        self.current_pose = msg.data.strip()
        with self.lock:
            if self.current_pose == self.current_goal:
                self.get_logger().info(f'[MissionPlanner] Arrived at {self.current_goal}')
                if self.timer:
                    self.timer.cancel()
                    self.timer = None

                if self.is_returning:
                    self.get_logger().info('[MissionPlanner] Return mission complete. Shutting down.')
                    rclpy.shutdown()
                    return

                self.current_index += 1
                self.publish_next_goal()

    def publish_next_goal(self):
        if self.current_index < len(self.goal_sequence):
            next_goal = self.goal_sequence[self.current_index]
            self.current_goal = next_goal
            self.start_goal_timer(next_goal)
        else:
            self.get_logger().info('[MissionPlanner] All goals completed.')
            time.sleep(5) 
            
    def start_goal_timer(self, goal_id: str):
        if self.timer:
            self.timer.cancel()
        self.timer = self.create_timer(1.0, lambda: self.publish_goal(goal_id))

    def publish_goal(self, goal_id: str):
        self.goal_pub.publish(String(data=goal_id))
        self.get_logger().info(f'[MissionPlanner] Published goal_pose: {goal_id}')

    def send_beep(self):
        msg = AudioNoteVector()
        msg.append = False
        msg.notes = [
            AudioNote(frequency=880, max_runtime=Duration(sec=0, nanosec=300_000_000)),  # 삐
            AudioNote(frequency=440, max_runtime=Duration(sec=0, nanosec=300_000_000)),  # 뽀
            AudioNote(frequency=880, max_runtime=Duration(sec=0, nanosec=300_000_000)),  # 삐
            AudioNote(frequency=440, max_runtime=Duration(sec=0, nanosec=300_000_000)),  # 뽀
        ]
        self.audio_pub.publish(msg)
        self.get_logger().info('[MissionPlanner] 삐뽀삐보 소리 전송 완료.')

def main(args=None):
    rclpy.init(args=args)
    node = MissionPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
