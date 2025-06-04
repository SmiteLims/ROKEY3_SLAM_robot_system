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

        # 로봇 ID 목록 (필요 시 확장 가능)
        self.robot_ids = ['robot0', 'robot1']

        # 로봇별 퍼블리셔 저장용 딕셔너리
        self.alert_pubs = {}

        # 각 로봇별로 status 구독 및 system_alert 퍼블리셔 생성
        for robot_id in self.robot_ids:
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

        # 이미 퍼블리시한 차량 ID 추적
        self.published_ids = set()

        self.get_logger().info("SystemController node started with multi-robot topic support")

    # 로봇별 status 콜백
    def status_callback(self, robot_id, msg: Int32):
        if msg.data == 1:
            alert = Int32()
            alert.data = 1
            self.alert_pubs[robot_id].publish(alert)
            self.get_logger().info(
                f"[{robot_id}] Received 1 → Published 1 to /{robot_id}/system_alert"
            )

    # 차량 정보 콜백
    def vehicle_info_callback(self, msg: VehicleInfo):
        task_id = msg.id
        location = msg.location

        if task_id in self.published_ids:
            return

        try:
            # FastAPI 호출로 DB에 task_id 존재 여부 확인
            api_url = f"http://localhost:8000/tasks/{task_id}/exists"
            res = requests.get(api_url)
            res.raise_for_status()
            exists = res.json().get("exists", False)

            if not exists:
                # DB에 없는 차량이면 경고 퍼블리시
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
