cmake_minimum_required(VERSION 3.8)
project(data_preprocessing)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

# add_executable
add_executable(camera_preprocessing src/camera_preprocessing.cpp)
ament_target_dependencies(camera_preprocessing rclcpp sensor_msgs)

add_executable(scan_downsampler_node src/scan_downsampler_node.cpp)
ament_target_dependencies(scan_downsampler_node rclcpp sensor_msgs)

# install
install(TARGETS
  camera_preprocessing
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  scan_downsampler_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
