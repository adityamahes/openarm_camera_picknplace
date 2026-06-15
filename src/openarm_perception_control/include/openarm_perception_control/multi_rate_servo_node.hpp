#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

class MultiRateServoNode : public rclcpp::Node
{
public:
  explicit MultiRateServoNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  void init();

private:
  void on_pick_prompt(const std_msgs::msg::String::SharedPtr msg);
  void on_pick_target(const geometry_msgs::msg::PointStamped::SharedPtr msg);

  bool move_to_scan_pose();
  bool wait_for_target(geometry_msgs::msg::PointStamped & target_out);
  bool move_to_pick_pose(const geometry_msgs::msg::PointStamped & target);
  geometry_msgs::msg::PoseStamped build_pick_pose(
    const geometry_msgs::msg::PointStamped & target) const;

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::CallbackGroup::SharedPtr cbg_prompt_;
  rclcpp::CallbackGroup::SharedPtr cbg_target_;

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr prompt_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr segment_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr target_sub_;

  std::mutex target_mutex_;
  std::condition_variable target_cv_;
  std::optional<geometry_msgs::msg::PointStamped> pending_target_;
  bool expecting_target_{false};

  std::vector<double> scan_joint_values_;
  double approach_height_m_;
  double planning_timeout_s_;
};
