#include <rclcpp/rclcpp.hpp>
#include "openarm_perception_control/multi_rate_servo_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<MultiRateServoNode>();
  node->init();

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
