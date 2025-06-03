import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Int32
from vehicle_info.msg import VehicleInfo
from vehicle_alert.msg import VehicleAlert
import requests
from functools import partial


class SystemController(Node):
    def __init__(self):
        super().__init__('system_controller')

        # 로봇 ID 목록 (확장 가능)
        self.robot_ids = ['robot0', 'robot1']

        # system_alert용 퍼블리셔 저장
        self.alert_pubs = {}

        # event → action 관련: 현재 값 저장 및 퍼블리셔 저장
        self.event_values = {robot_id: 0 for robot_id in self.robot_ids}
        self.action_pubs = {}

        # === 각 로봇에 대해 공통 설정 ===
        for robot_id in self.robot_ids:
            # /robotX/status 구독 → /robotX/system_alert 퍼블리시
            self.create_subscription(
                Int32,
                f'/{robot_id}/status',
                partial(self.status_callback, robot_id),
                10
            )
            self.alert_pubs[robot_id] = self.create_publisher(
                Int32,
                f'/{robot_id}/system_alert',
                10
            )

            # /robotX/event 구독 → /robotX/action 주기적 퍼블리시
            self.create_subscription(
                Int32,
                f'/{robot_id}/event',
                partial(self.event_callback, robot_id),
                10
            )
            self.action_pubs[robot_id] = self.create_publisher(
                Int32,
                f'/{robot_id}/action',
                10
            )

            # 1초마다 /robotX/action 퍼블리시
            self.create_timer(
                1,
                partial(self.publish_action, robot_id)
            )

        # vehicle_info 구독
        self.vehicle_sub = self.create_subscription(
            VehicleInfo,
            'vehicle_info',
            self.vehicle_info_callback,
            10
        )

        # vehicle_alert 퍼블리셔
        self.vehicle_alert_pub = self.create_publisher(
            VehicleAlert,
            'vehicle_alert',
            10
        )

        # 이미 알림 보낸 차량 ID 저장
        self.published_ids = set()

        self.get_logger().info("SystemController with EventAction support started")

    # /robotX/status 콜백: 1이 오면 /robotX/system_alert에 1 퍼블리시
    def status_callback(self, robot_id, msg: Int32):
        if msg.data == 1:
            alert = Int32()
            alert.data = 1
            self.alert_pubs[robot_id].publish(alert)
            self.get_logger().info(
                f"[{robot_id}] Received 1 → Published 1 to /{robot_id}/system_alert"
            )

    # /robotX/event 콜백: 값(0 또는 1)을 저장
    def event_callback(self, robot_id, msg: Int32):
        self.event_values[robot_id] = msg.data
        self.get_logger().info(f"[{robot_id}] Received event: {msg.data}")

    # 1초 주기 타이머 콜백: 저장된 값으로 /robotX/action 퍼블리시
    def publish_action(self, robot_id):
        msg = Int32()
        msg.data = self.event_values[robot_id]
        self.action_pubs[robot_id].publish(msg)
        # self.get_logger().info(f"[{robot_id}] Published action: {msg.data}")

    # vehicle_info 콜백: DB에 없는 차량이면 vehicle_alert 퍼블리시
    def vehicle_info_callback(self, msg: VehicleInfo):
        task_id = msg.id
        location = msg.location

        if task_id in self.published_ids:
            return

        try:
            api_url = f"http://localhost:8000/tasks/{task_id}/exists"
            res = requests.get(api_url)
            res.raise_for_status()
            exists = res.json().get("exists", False)

            if not exists:
                alert_msg = VehicleAlert()
                alert_msg.id = task_id
                alert_msg.location = location
                self.vehicle_alert_pub.publish(alert_msg)
                self.get_logger().info(
                    f"[ALERT] Task ID {task_id} not found → Published vehicle_alert"
                )
                self.published_ids.add(task_id)

        except Exception as e:
            self.get_logger().error(f"API call failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SystemController()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
