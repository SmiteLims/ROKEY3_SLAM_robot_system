#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <cmath>
#include <vector>

using std::placeholders::_1;

class ScanDownsampler : public rclcpp::Node
{
public:
  ScanDownsampler()
  : Node("scan_downsampler")
  {
    rclcpp::SubscriptionOptions options;
    options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

    // 구독자 생성 (/scan)
    subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "/robot0/scan",
      rclcpp::SensorDataQoS(),
      std::bind(&ScanDownsampler::scan_callback, this, _1),
      options);

    // 퍼블리셔 생성 (/scan_downsampled)
    scan_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>(
      "/robot0/scan/downsample", 10);

    RCLCPP_INFO(this->get_logger(), "ScanDownsampler Node Initialized.");
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_pub_;

  void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
  {
    int step = 5; // 다운샘플링 간격

    auto downsampled_msg = sensor_msgs::msg::LaserScan();
    downsampled_msg.header = msg->header;
    downsampled_msg.angle_min = msg->angle_min;
    downsampled_msg.angle_max = msg->angle_max;
    downsampled_msg.angle_increment = msg->angle_increment * step;
    downsampled_msg.time_increment = msg->time_increment * step;
    downsampled_msg.scan_time = msg->scan_time;
    downsampled_msg.range_min = msg->range_min;
    downsampled_msg.range_max = msg->range_max;

    // ranges, intensities 다운샘플링
    for (size_t i = 0; i < msg->ranges.size(); i += step)
    {
      downsampled_msg.ranges.push_back(msg->ranges[i]);
      if (i < msg->intensities.size())
        downsampled_msg.intensities.push_back(msg->intensities[i]);
    }

    scan_pub_->publish(downsampled_msg);
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ScanDownsampler>());
  rclcpp::shutdown();
  return 0;
}
