#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import math
import time

# action: 가장 빠른 전달
qos_action = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    depth=1
)

# === (1) quaternion_from_euler 직접 구현 ===
def quaternion_from_euler(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (x, y, z, w)

# === (2) 동서남북 매핑 ===
DIRECTION_TO_YAW = {
    'E': 0.0,
    'N': math.pi / 2,
    'W': math.pi,
    'S': -math.pi / 2
}

# === (3) 이동할 Waypoints: (x, y, direction) ===
WAYPOINTS = [
    (5.428, 1.885, 'W'),
    (4.668, 1.885, 'W'),
    (4.668, 1.305, 'S'),
    (5.398, 0.740, 'E'),
    (4.668, 1.305, 'N'),
    (4.668, 1.885, 'E'),
    (5.428, 1.885, 'E')
]

# === (4) 방향 문자열을 Yaw 각도로 변환 ===
def direction_to_yaw(direction):
    if direction not in DIRECTION_TO_YAW:
        raise ValueError(f"Invalid direction: {direction}. Use 'N', 'E', 'S', or 'W'.")
    return DIRECTION_TO_YAW[direction]

# === (5) Pose 생성 함수 ===
def create_pose(x, y, yaw_rad, navigator):
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x = x
    pose.pose.position.y = y

    q = quaternion_from_euler(0, 0, yaw_rad)
    pose.pose.orientation.x = q[0]
    pose.pose.orientation.y = q[1]
    pose.pose.orientation.z = q[2]
    pose.pose.orientation.w = q[3]
    return pose

# === (6) Main Node ===
class MissionExecutor(Node):
    def __init__(self):
        super().__init__('mission_executor')

        # Navigators
        self.dock_navigator = TurtleBot4Navigator()
        self.nav_navigator = BasicNavigator(node_name='navigator_robot1')

        # Mission Start Flag
        self.mission_started = False

        # Action Control Flag (0: run, 1: stop)
        self.action_status = 0

        # Subscriber: Mission Start
        self.start_sub = self.create_subscription(
            Int32,
            '/robot1/system_alert',    # ★ 미션 시작 신호
            self.start_mission_callback,
            10 #qos_system_alert
        )

        # Subscriber: Action Control
        self.action_sub = self.create_subscription(
            Int32,
            '/robot1/action',    # ★ 추가된 subscriber
            self.action_control_callback,
            qos_action
        )

    def start_mission_callback(self, msg):
        if msg.data == 1 and not self.mission_started:
            self.get_logger().info('1 수신: 미션 시작!')
            self.mission_started = True

    def action_control_callback(self, msg):
        if msg.data == 0:
            if self.action_status != 0:
                self.get_logger().info('액션 제어: 재개 (0 수신)')
            self.action_status = 0
        elif msg.data == 1:
            if self.action_status != 1:
                self.get_logger().info('액션 제어: 정지 (1 수신)')
            self.action_status = 1
        else:
            self.get_logger().warn(f'알 수 없는 액션 값 수신: {msg.data}')

    def wait_until_nav2_ready(self):
        self.nav_navigator.waitUntilNav2Active()

    def run(self):
        # (1) 초기 위치 설정
        initial_pose = create_pose(6.188, 1.885, -0.004, self.nav_navigator)
        self.nav_navigator.setInitialPose(initial_pose)
        self.nav_navigator.get_logger().info(f'초기 위치 설정 중...')
        time.sleep(2.0)  # AMCL 및 TF 안정화 대기
        self.wait_until_nav2_ready()

        # (2) 1 수신 대기
        self.get_logger().info('미션 시작 신호(1) 수신 대기 중...')
        while rclpy.ok() and not self.mission_started:
            rclpy.spin_once(self, timeout_sec=0.5)

        # (3) 도킹 상태면 언도킹
        if self.dock_navigator.getDockedStatus():
            self.dock_navigator.get_logger().info('도킹 상태 → 언도킹 시작')
            self.dock_navigator.undock()
            self.dock_navigator.get_logger().info('언도킹 완료')
        else:
            self.dock_navigator.get_logger().info('이미 언도킹 상태')

        # (4) Waypoints 이동
        for idx, wp in enumerate(WAYPOINTS):
            x, y, direction = wp
            yaw_rad = direction_to_yaw(direction)
            goal_pose = create_pose(x, y, yaw_rad, self.nav_navigator)

            # 목표 설정
            self.nav_navigator.goToPose(goal_pose)
            self.nav_navigator.get_logger().info(f'Waypoint {idx+1} ({direction})로 이동 시작')

            while not self.nav_navigator.isTaskComplete():
                feedback = self.nav_navigator.getFeedback()
                if feedback and feedback.current_pose:
                    current_x = feedback.current_pose.pose.position.x
                    current_y = feedback.current_pose.pose.position.y
                    self.nav_navigator.get_logger().info(f'현재 위치: ({current_x:.2f}, {current_y:.2f})')

                # 주행 중 action 상태 체크
                if self.action_status == 1:  # Stop 명령 수신
                    self.nav_navigator.cancelTask()
                    self.get_logger().info('주행 정지 요청 → 현재 위치에서 정지')
                    time.sleep(2.0)
                    # 재개 신호(0) 수신 대기
                    while rclpy.ok() and self.action_status == 1:
                        rclpy.spin_once(self, timeout_sec=0.5)
                    self.get_logger().info('주행 재개 요청 수신 → 다시 목표 설정')
                    self.nav_navigator.goToPose(goal_pose)

                if self.nav_navigator.isTaskComplete():
                    break

                rclpy.spin_once(self, timeout_sec=0.5)  # spin_once 추가

            # 결과 확인
            result = self.nav_navigator.getResult()
            if result == TaskResult.SUCCEEDED:
                self.nav_navigator.get_logger().info(f'Waypoint {idx+1} 도착! 2초 대기 중...')
                time.sleep(2.0)
            elif result == TaskResult.CANCELED:
                self.nav_navigator.get_logger().warn(f'Waypoint {idx+1} 이동이 취소되었습니다.')
            elif result == TaskResult.FAILED:
                error_code, error_msg = self.nav_navigator.getTaskError()
                self.nav_navigator.get_logger().error(f'Waypoint {idx+1} 이동 실패: {error_code} - {error_msg}')
            else:
                self.nav_navigator.get_logger().warn('알 수 없는 결과 코드 수신')

        # (5) 이동 완료 후 도킹
        self.nav_navigator.get_logger().info('모든 waypoint 도착 완료. 도킹 시작')
        self.dock_navigator.dock()
        self.dock_navigator.get_logger().info('도킹 요청 완료')

        # (6) 노드 종료
        self.dock_navigator.destroy_node()
        self.nav_navigator.destroy_node()

# === (7) 프로그램 시작 ===
def main():
    rclpy.init()
    executor = MissionExecutor()
    executor.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()