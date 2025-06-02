import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor  # 병렬 콜백 처리를 위한 executor
from std_msgs.msg import Int32                     # 정수 메시지 타입 (표준)
from vehicle_info.msg import VehicleInfo, VehicleAlert  # 커스텀 메시지 타입 (차량 정보 및 경고)
import requests  # FastAPI API 호출을 위한 HTTP 라이브러리


class SystemController(Node):
    def __init__(self):
        super().__init__('system_controller')  # 노드 이름 정의

        # /status 토픽 구독 (CCTV에서 객체가 감지되었는지 확인)
        self.status_sub = self.create_subscription( ############ robot0,1 구분 필요
            Int32,                # 메시지 타입
            '/status',           # 구독할 토픽 이름
            self.status_callback,# 콜백 함수
            10                   # QoS: 큐 사이즈
        )

        # /system_alert 토픽 퍼블리셔 (순찰 로봇에 명령 내림)
        self.alert_pub = self.create_publisher( ############ robot0,1 구분 필요
            Int32,                # 메시지 타입
            '/system_alert',      # 퍼블리시할 토픽 이름
            10
        )

        # /vehicle_info 토픽 구독 (차량 번호 및 위치 정보 수신)
        self.vehicle_sub = self.create_subscription(
            VehicleInfo,          # 메시지 타입 (id, location)
            'vehicle_info',       # 구독할 토픽
            self.vehicle_info_callback,
            10
        )

        # /vehicle_alert 토픽 퍼블리셔 (DB에 등록되지 않은 차량을 알림)
        self.vehicle_alert_pub = self.create_publisher(
            VehicleAlert,         # 메시지 타입 (id, location)
            'vehicle_alert',      # 퍼블리시할 토픽
            10
        )

        # 이미 경고를 보낸 차량 ID는 다시 퍼블리시하지 않도록 기록
        self.published_ids = set()

        self.get_logger().info("SystemController node started with multi-topic support")

    # /status 토픽 수신 시 호출되는 콜백
    def status_callback(self, msg: Int32): ############ robot0,1 구분 필요
        if msg.data == 1:  # 객체 감지됨
            alert = Int32()
            alert.data = 1
            self.alert_pub.publish(alert)  # 순찰 명령 전송
            self.get_logger().info("Received 1 → Published 1 to /system_alert")

    # /vehicle_info 토픽 수신 시 호출되는 콜백
    def vehicle_info_callback(self, msg: VehicleInfo):
        task_id = msg.id
        location = msg.location

        # 이미 퍼블리시한 차량이라면 중복 알림 방지
        if task_id in self.published_ids:
            return

        try:
            # FastAPI로 task_id 존재 여부 확인 요청
            api_url = f"http://localhost:8000/tasks/{task_id}/exists"
            res = requests.get(api_url)
            res.raise_for_status()  # HTTP 에러가 있으면 예외 발생
            data = res.json()
            exists = data.get("exists", False)

            if not exists:
                # DB에 없는 차량일 경우 vehicle_alert 퍼블리시
                alert_msg = VehicleAlert()
                alert_msg.id = task_id
                alert_msg.location = location
                self.vehicle_alert_pub.publish(alert_msg)

                self.get_logger().info(
                    f"[ALERT] Task ID {task_id} not found → Published alert")

                # 중복 퍼블리시 방지용으로 기록
                self.published_ids.add(task_id)

        except Exception as e:
            # FastAPI 호출 실패 시 에러 로그 출력
            self.get_logger().error(f"API call failed: {e}")

    ########### Event 발생 상황 추가 (예정) #############


def main(args=None):
    rclpy.init(args=args)

    node = SystemController()

    # 병렬 콜백 처리를 위한 MultiThreadedExecutor 사용
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()  # 노드 실행 (콜백 처리 loop)
    finally:
        node.destroy_node()  # 종료 시 노드 제거
        rclpy.shutdown()     # ROS 종료


if __name__ == '__main__':
    main()
