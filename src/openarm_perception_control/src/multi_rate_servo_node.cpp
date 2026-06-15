#include "openarm_perception_control/multi_rate_servo_node.hpp"

#include <chrono>
#include <tf2/LinearMath/Quaternion.h>

using namespace std::chrono_literals;

MultiRateServoNode::MultiRateServoNode(const rclcpp::NodeOptions & options)
: Node("multi_rate_servo_node", options)
{
  this->declare_parameter(
    "scan_joint_values",
    std::vector<double>{0.0, -0.3, 0.0, 1.6, 0.0, 1.57, 0.0});
  this->declare_parameter("approach_height", 0.15);
  this->declare_parameter("planning_timeout", 10.0);

  scan_joint_values_  = this->get_parameter("scan_joint_values").as_double_array();
  approach_height_m_  = this->get_parameter("approach_height").as_double();
  planning_timeout_s_ = this->get_parameter("planning_timeout").as_double();

  tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  cbg_prompt_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  cbg_target_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  rclcpp::SubscriptionOptions prompt_opts;
  prompt_opts.callback_group = cbg_prompt_;
  prompt_sub_ = this->create_subscription<std_msgs::msg::String>(
    "/pick_prompt", 10,
    std::bind(&MultiRateServoNode::on_pick_prompt, this, std::placeholders::_1),
    prompt_opts);

  segment_pub_ = this->create_publisher<std_msgs::msg::String>("/segment_prompt", 10);

  rclcpp::SubscriptionOptions target_opts;
  target_opts.callback_group = cbg_target_;
  target_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
    "/pick_target", 10,
    std::bind(&MultiRateServoNode::on_pick_target, this, std::placeholders::_1),
    target_opts);
}

void MultiRateServoNode::init()
{
  move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
    shared_from_this(), "right_arm");
  move_group_->setPlanningTime(planning_timeout_s_);
  move_group_->setMaxVelocityScalingFactor(0.3);
  move_group_->setMaxAccelerationScalingFactor(0.1);
  RCLCPP_INFO(this->get_logger(), "Initialized. Planning frame: %s",
    move_group_->getPlanningFrame().c_str());
}

void MultiRateServoNode::on_pick_prompt(const std_msgs::msg::String::SharedPtr msg)
{
  RCLCPP_INFO(this->get_logger(), "Pick request: '%s'", msg->data.c_str());

  if (!move_to_scan_pose()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to reach scan pose");
    return;
  }

  rclcpp::sleep_for(500ms);

  auto prompt_msg = std_msgs::msg::String();
  prompt_msg.data = msg->data;
  segment_pub_->publish(prompt_msg);

  geometry_msgs::msg::PointStamped target;
  if (!wait_for_target(target)) {
    RCLCPP_ERROR(this->get_logger(), "Segmentation timed out");
    return;
  }

  if (!move_to_pick_pose(target)) {
    RCLCPP_ERROR(this->get_logger(), "Failed to reach pick pose");
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Pick complete");
}

void MultiRateServoNode::on_pick_target(const geometry_msgs::msg::PointStamped::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(target_mutex_);
  if (expecting_target_) {
    pending_target_ = *msg;
    target_cv_.notify_one();
  }
}

bool MultiRateServoNode::move_to_scan_pose()
{
  move_group_->setJointValueTarget(scan_joint_values_);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  if (move_group_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(this->get_logger(), "Planning to scan pose failed");
    return false;
  }
  if (move_group_->execute(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(this->get_logger(), "Executing scan pose failed");
    return false;
  }
  RCLCPP_INFO(this->get_logger(), "Reached scan pose");
  return true;
}

bool MultiRateServoNode::wait_for_target(geometry_msgs::msg::PointStamped & target_out)
{
  {
    std::lock_guard<std::mutex> lock(target_mutex_);
    expecting_target_ = true;
    pending_target_.reset();
  }

  std::unique_lock<std::mutex> lock(target_mutex_);
  bool got = target_cv_.wait_for(lock, std::chrono::seconds(30),
    [this] { return pending_target_.has_value(); });
  expecting_target_ = false;

  if (!got) {
    RCLCPP_ERROR(this->get_logger(), "Timed out waiting for segmentation result");
    return false;
  }

  const std::string planning_frame = move_group_->getPlanningFrame();
  try {
    target_out = tf_buffer_->transform(*pending_target_, planning_frame,
      tf2::durationFromSec(1.0));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR(this->get_logger(), "TF transform failed: %s", ex.what());
    return false;
  }

  RCLCPP_INFO(this->get_logger(), "Target in '%s': (%.3f, %.3f, %.3f) m",
    planning_frame.c_str(),
    target_out.point.x, target_out.point.y, target_out.point.z);
  return true;
}

bool MultiRateServoNode::move_to_pick_pose(const geometry_msgs::msg::PointStamped & target)
{
  auto pick_pose    = build_pick_pose(target);
  auto approach_pose = pick_pose;
  approach_pose.pose.position.z += approach_height_m_;

  moveit::planning_interface::MoveGroupInterface::Plan plan;

  move_group_->setPoseTarget(approach_pose);
  if (move_group_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(this->get_logger(), "Planning to approach pose failed");
    return false;
  }
  if (move_group_->execute(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(this->get_logger(), "Executing approach pose failed");
    return false;
  }

  move_group_->setPoseTarget(pick_pose);
  if (move_group_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(this->get_logger(), "Planning to pick pose failed");
    return false;
  }
  if (move_group_->execute(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(this->get_logger(), "Executing pick pose failed");
    return false;
  }

  RCLCPP_INFO(this->get_logger(), "Reached pick pose");
  return true;
}

geometry_msgs::msg::PoseStamped MultiRateServoNode::build_pick_pose(
  const geometry_msgs::msg::PointStamped & target) const
{
  geometry_msgs::msg::PoseStamped pose;
  pose.header        = target.header;
  pose.pose.position = target.point;

  tf2::Quaternion q;
  q.setRPY(M_PI, 0.0, 0.0);
  q.normalize();
  pose.pose.orientation.x = q.x();
  pose.pose.orientation.y = q.y();
  pose.pose.orientation.z = q.z();
  pose.pose.orientation.w = q.w();

  return pose;
}
