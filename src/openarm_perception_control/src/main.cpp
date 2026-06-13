/*
 * main.cpp
 * ========
 * Entry point for the openarm_perception_control node.
 *
 * Startup sequence:
 *   1. rclcpp::init — initialise the ROS 2 client library.
 *   2. make_shared<MultiRateServoNode> — constructor runs: declares params,
 *      creates TF listener, service client, and service server.
 *   3. node->init() — deferred step that creates MoveGroupInterface
 *      (requires shared_from_this(), so must happen after make_shared).
 *   4. MultiThreadedExecutor::spin() — allows the /pick_object handler and
 *      the /segment_object response callback to run on separate threads,
 *      avoiding the deadlock that would occur with a single-threaded executor.
 */

#include <rclcpp/rclcpp.hpp>
#include "openarm_perception_control/multi_rate_servo_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<MultiRateServoNode>();

  // init() must be called after make_shared so shared_from_this() is valid
  node->init();

  // MultiThreadedExecutor is required — see multi_rate_servo_node.hpp for why
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
