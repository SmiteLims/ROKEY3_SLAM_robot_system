#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator

class DockController(Node):
    def __init__(self):
        super().__init__('dock_controller1')

        # 1) TurtleBot4Navigator 인스턴스 생성
        self.dock_navigator = TurtleBot4Navigator()

        # 2) 현재 state_pose, goal_pose 초기화
        self.state_pose = None
        self.goal_pose = None

        # 3) 구독: /bfs/state_pose, /bfs/goal_pose
        self.create_subscription(
            String,
            '/robot1/bfs/state_pose',
            self.state_callback,
            10
        )
        self.create_subscription(
            String,
            '/robot1/bfs/goal_pose',
            self.goal_callback,
            10
        )

        # 4) 0.5초 주기로 도킹/언도킹 로직을 확인하는 타이머
        self.timer = self.create_timer(0.5, self.timer_callback)

        self.get_logger().info('[DockController1] Initialized and waiting for state_pose & goal_pose...')

    def state_callback(self, msg: String):
        self.state_pose = msg.data
        self.get_logger().info(f'[DockController1] state_pose <- "{self.state_pose}"')

    def goal_callback(self, msg: String):
        self.goal_pose = msg.data
        self.get_logger().info(f'[DockController1] goal_pose  <- "{self.goal_pose}"')

    def timer_callback(self):
        # state_pose와 goal_pose가 모두 수신된 이후에만 처리
        if self.state_pose is None or self.goal_pose is None:
            return

        # 1) state_pose == "p1_0" and goal_pose == "p1_0" --> 무조건 도킹
        if self.state_pose == 'p1_0' and self.goal_pose == 'p1_0':
            # 이미 도킹 중인지 확인
            if not self.dock_navigator.getDockedStatus():
                self.get_logger().info('[DockController1] 조건 충족: state==p1_0 & goal==p1_0 → dock() 호출')
                self.dock_navigator.dock()
            else:
                # 이미 도킹되어 있다면 아무 동작도 하지 않음
                self.get_logger().debug('[DockController1] 이미 도킹 상태입니다.')

        # 2) state_pose == "p1_0" and goal_pose != "p1_0" and (현재 도킹 상태라면) --> 언도킹
        elif self.state_pose == 'init_pose' and self.goal_pose != 'p1_0':
            if self.dock_navigator.getDockedStatus():
                self.get_logger().info('[DockController1] 조건 충족: state==p1_0 & goal!=p1_0 & docked → undock() 호출')
                self.dock_navigator.undock()
            else:
                self.get_logger().debug('[DockController1] 도킹 상태가 아닙니다. 언도킹 필요 없음.')

        # 그 외의 경우에는 도킹/언도킹 동작하지 않음
        else:
            self.get_logger().debug('[DockController1] 도킹/언도킹 조건에 해당하지 않습니다.')


def main(args=None):
    rclpy.init(args=args)
    node = DockController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
