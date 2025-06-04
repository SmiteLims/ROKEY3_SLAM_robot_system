import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

class SystemController(Node):
    def __init__(self):
        super().__init__('system_controller')

        # /status 구독 # CCTV의 객체 감지
        self.subscription = self.create_subscription(
            Int32,
            '/status',
            self.status_callback,
            10)

        # /system_alert 퍼블리셔 (순찰 명령)
        self.alert_publisher = self.create_publisher(Int32, '/system_alert', 10)

        self.get_logger().info("System Controller Node Started")

    def status_callback(self, msg: Int32):
        if msg.data == 1:
            alert = Int32()
            alert.data = 1
            self.alert_publisher.publish(alert)
            self.get_logger().info("Received 1 → Published 1 to /system_alert")

            

def main(args=None):
    rclpy.init(args=args)
    node = SystemController()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
