#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"

class ImageRelayNode : public rclcpp::Node
{
public:
  explicit ImageRelayNode(const rclcpp::NodeOptions & options)
  : Node("camera_preprocessing", options)
  {
    // ✅ Fastest QoS: Best effort, keep last 1 (sensor data)
    rclcpp::QoS qos_profile(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data));
    qos_profile.best_effort();
    qos_profile.keep_last(1);

    // ✅ RGB Subscriber
    rgb_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
      "/robot0/oakd/rgb/image_raw/compressed",
      qos_profile,
      std::bind(&ImageRelayNode::rgbCallback, this, std::placeholders::_1)
    );

    // ✅ Depth Subscriber
    depth_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
      "/robot0/oakd/stereo/image_raw/compressedDepth",
      qos_profile,
      std::bind(&ImageRelayNode::depthCallback, this, std::placeholders::_1)
    );

    // ✅ RGB Publisher
    rgb_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
      "/robot0/oakd/rgb/image_fast/compressed",
      qos_profile
    );

    // ✅ Depth Publisher
    depth_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
      "/robot0/oakd/stereo/image_fast/compressedDepth",
      qos_profile
    );

    RCLCPP_INFO(this->get_logger(), "📷 Image relay node initialized with zero-copy & fast QoS.");
  }

private:
  void rgbCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
  {
    rgb_pub_->publish(*msg);
  }

  void depthCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
  {
    depth_pub_->publish(*msg);
  }

  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr rgb_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr depth_sub_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr rgb_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr depth_pub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true);  // ✅ Zero-copy 활성화

  rclcpp::spin(std::make_shared<ImageRelayNode>(options));
  rclcpp::shutdown();
  return 0;
}

