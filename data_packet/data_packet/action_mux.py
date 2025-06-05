#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Bool
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

class ActionMux(Node):
    def __init__(self, robot_ns):
        super().__init__(f'{robot_ns}_action_mux')
        self.robot_ns = robot_ns

        self.priority = {
            'dashboard': 1,
            'auto': 0,
        }
        self.sources = list(self.priority.keys())

        self.active_source = None
        self.last_cmd_time = self.get_clock().now()
        self.source_last_msg = {src: None for src in self.sources}
        self.source_last_time = {src: self.get_clock().now() for src in self.sources}
        self.override_active = False

        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.pub = self.create_publisher(Int32, f'/{robot_ns}/action', qos)
        self.override_pub = self.create_publisher(Bool, f'/{robot_ns}/action_override', qos)

        # 동적으로 콜백 등록 (여기선 dashboard/auto만)
        for src in self.sources:
            topic = f'/{robot_ns}/action_{src}'
            self.create_subscription(Int32, topic, self._make_cb(src), qos)

        # 타임아웃 체크
        self.create_timer(0.2, self.check_timeout)

    def _make_cb(self, src):
        def cb(msg):
            self.source_last_msg[src] = msg
            self.source_last_time[src] = self.get_clock().now()

            # dashboard에서 STOP(1) 시 override ON (가장 높은 우선순위)
            if src == 'dashboard' and msg.data == 1:
                if not self.override_active:
                    self.get_logger().info("Manual override ON (STOP): 모든 source 무시")
                self.override_active = True
                self.active_source = 'dashboard'
                self.pub.publish(msg)
                self.override_pub.publish(Bool(data=True))
                return

            # dashboard에서 START(0) 시 override 해제
            if src == 'dashboard' and msg.data == 0:
                if self.override_active:
                    self.get_logger().info("Manual override OFF (START): 우선순위 재선택")
                self.override_active = False
                self.active_source = None
                self.override_pub.publish(Bool(data=False))
                self.publish_highest_priority()
                return

            # override 중이면 dashboard만 허용
            if self.override_active:
                return

            prev = self.active_source
            top_src = self.select_highest_priority_source()
            self.active_source = top_src
            if src == top_src:
                self.pub.publish(msg)
            if prev != self.active_source:
                self.get_logger().info(f"Active source: {self.active_source}")
        return cb

    def select_highest_priority_source(self):
        # 최근 들어온 source들 중, 우선순위 가장 높은 것 선택
        active = [(src, self.source_last_time[src]) for src in self.sources if self.source_last_msg[src] is not None]
        if not active:
            return None
        # 우선순위 높은 순 정렬
        top = sorted(active, key=lambda x: self.priority[x[0]], reverse=True)
        return top[0][0]

    def publish_highest_priority(self):
        src = self.select_highest_priority_source()
        if src and self.source_last_msg[src]:
            self.pub.publish(self.source_last_msg[src])

    def check_timeout(self):
        if self.override_active:
            return
        if self.active_source:
            now = self.get_clock().now()
            dt = (now - self.source_last_time[self.active_source]).nanoseconds
            if dt > 1e9:
                self.get_logger().info(f"{self.active_source} input timeout. Stopping.")
                self.source_last_msg[self.active_source] = None
                self.active_source = None
                stop_msg = Int32()
                stop_msg.data = 1
                self.pub.publish(stop_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ActionMux('robot0')  # robot_ns만 바꿔서 여러 노드 띄우면 됨
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
