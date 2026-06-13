/*
 * multi_rate_servo_node.cpp
 * =========================
 * Implementation of the arm-orchestration pipeline for perception-guided
 * pick-and-place.
 *
 * Node flow (triggered by a /pick_object service call):
 *
 *   [1] move_to_scan_pose()
 *         MoveIt plans and executes a trajectory to the overhead scan pose
 *         so the wrist-mounted RealSense camera looks straight down.
 *
 *   [2] call_segmentation()
 *         Sends the user's text prompt to the /segment_object service
 *         (openarm_sam_perception), waits for the 3D centroid, then
 *         transforms it from the camera optical frame into the MoveIt
 *         planning frame using TF2.
 *
 *   [3] move_to_pick_pose()
 *         MoveIt moves the gripper to an approach pose (above the target),
 *         then to the pick pose (at the target, gripper pointing down).
 */

#include "openarm_perception_control/multi_rate_servo_node.hpp"

#include <chrono>
#include <tf2/LinearMath/Quaternion.h>

using namespace std::chrono_literals;
using SegmentObject = openarm_perception_msgs::srv::SegmentObject;
using PickObject    = openarm_perception_msgs::srv::PickObject;

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

MultiRateServoNode::MultiRateServoNode(const rclcpp::NodeOptions & options)
: Node("multi_rate_servo_node", options)
{
  /*
   * Declare all tunable parameters.
   * Default scan_joint_values position the arm so the camera faces
   * roughly downward — calibrate these with the real robot before use.
   */
  this->declare_parameter(
    "scan_joint_values",
    std::vector<double>{0.0, -0.3, 0.0, 1.6, 0.0, 1.57, 0.0});
  this->declare_parameter("approach_height", 0.15);   // metres above pick point
  this->declare_parameter("planning_timeout", 10.0);  // MoveIt planning budget (s)

  scan_joint_values_  = this->get_parameter("scan_joint_values").as_double_array();
  approach_height_m_  = this->get_parameter("approach_height").as_double();
  planning_timeout_s_ = this->get_parameter("planning_timeout").as_double();

  // TF2 buffer + listener — needed to transform the camera-frame point
  tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  /*
   * Two MutuallyExclusive callback groups on a MultiThreadedExecutor:
   *
   *   cbg_pick_service_  — runs handle_pick_object (long-running; blocks
   *                         through arm motion + segmentation + arm motion).
   *   cbg_seg_client_    — runs the /segment_object response callback.
   *
   * Keeping them on separate groups lets the executor dispatch them on
   * independent threads, preventing the pick handler from blocking the
   * segmentation response and causing a deadlock.
   */
  cbg_pick_service_ =
    this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  cbg_seg_client_ =
    this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  // Client for the Python perception node's /segment_object service
  seg_client_ = this->create_client<SegmentObject>(
    "segment_object",
    rmw_qos_profile_services_default,
    cbg_seg_client_);   // responses land on cbg_seg_client_

  // Server that external callers use to trigger the full pipeline
  pick_service_ = this->create_service<PickObject>(
    "pick_object",
    std::bind(
      &MultiRateServoNode::handle_pick_object, this,
      std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default,
    cbg_pick_service_);  // handler runs on cbg_pick_service_
}

// ---------------------------------------------------------------------------
// init() — deferred MoveGroupInterface construction
// ---------------------------------------------------------------------------

void MultiRateServoNode::init()
{
  /*
   * MoveGroupInterface requires a shared_ptr to the node, which is only
   * available after make_shared() returns.  Call this once from main().
   */
  move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
    shared_from_this(), "right_arm");

  // Limit speed so arm motion is safe around the workspace
  move_group_->setPlanningTime(planning_timeout_s_);
  move_group_->setMaxVelocityScalingFactor(0.3);      // 30 % of joint speed limits
  move_group_->setMaxAccelerationScalingFactor(0.1);  // 10 % of joint accel limits

  RCLCPP_INFO(this->get_logger(),
    "Initialized. Planning group: right_arm  |  Planning frame: %s",
    move_group_->getPlanningFrame().c_str());
}

// ---------------------------------------------------------------------------
// Step 0 — /pick_object service handler (top-level pipeline)
// ---------------------------------------------------------------------------

void MultiRateServoNode::handle_pick_object(
  std::shared_ptr<PickObject::Request> request,
  std::shared_ptr<PickObject::Response> response)
{
  RCLCPP_INFO(this->get_logger(), "Pick request: '%s'", request->text_prompt.c_str());

  // Step 1: move arm to overhead scan position
  if (!move_to_scan_pose()) {
    response->success = false;
    response->message = "Failed to reach scan pose";
    return;
  }

  // Brief pause so camera motion blur settles before capturing a frame
  rclcpp::sleep_for(500ms);

  // Step 2: ask perception node to find and locate the object
  geometry_msgs::msg::PointStamped target;
  if (!call_segmentation(request->text_prompt, target)) {
    response->success = false;
    response->message = "Segmentation failed";
    return;
  }

  // Step 3: move gripper above the object then descend to pick
  if (!move_to_pick_pose(target)) {
    response->success = false;
    response->message = "Failed to reach pick pose";
    return;
  }

  response->success        = true;
  response->message        = "Pick complete";
  response->final_position = target;
}

// ---------------------------------------------------------------------------
// Step 1 — move to scan pose
// ---------------------------------------------------------------------------

bool MultiRateServoNode::move_to_scan_pose()
{
  // Set all 7 right-arm joints to the configured overhead scan values
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

// ---------------------------------------------------------------------------
// Step 2 — call /segment_object and transform result to planning frame
// ---------------------------------------------------------------------------

bool MultiRateServoNode::call_segmentation(
  const std::string & text_prompt,
  geometry_msgs::msg::PointStamped & target_out)
{
  if (!seg_client_->wait_for_service(5s)) {
    RCLCPP_ERROR(this->get_logger(), "segment_object service unavailable");
    return false;
  }

  auto req = std::make_shared<SegmentObject::Request>();
  req->text_prompt = text_prompt;

  /*
   * Why std::promise instead of spin_until_future_complete?
   *
   * handle_pick_object runs as a ROS callback on cbg_pick_service_.
   * spin_until_future_complete would try to spin the node a second time
   * from inside an active callback — this causes a deadlock.
   *
   * Instead:
   *   1. async_send_request sends the request and returns immediately.
   *   2. The lambda captures a shared_ptr to a std::promise.
   *      When the MultiThreadedExecutor delivers the response on
   *      cbg_seg_client_ (a different thread), the lambda fulfills
   *      the promise.
   *   3. This function blocks on future.wait_for() while the executor
   *      stays free to process the response callback.
   */
  auto promise = std::make_shared<std::promise<SegmentObject::Response::SharedPtr>>();
  std::future<SegmentObject::Response::SharedPtr> fut = promise->get_future();

  seg_client_->async_send_request(
    req,
    [promise](rclcpp::Client<SegmentObject>::SharedFuture shared_fut) {
      promise->set_value(shared_fut.get());
    });

  // Wait up to 30 s for GroundingDINO + MobileSAM to finish
  if (fut.wait_for(30s) != std::future_status::ready) {
    RCLCPP_ERROR(this->get_logger(), "segment_object service timed out");
    return false;
  }

  auto res = fut.get();
  if (!res->success) {
    RCLCPP_ERROR(this->get_logger(), "Segmentation: %s", res->message.c_str());
    return false;
  }

  /*
   * The point comes back in the camera optical frame (frame_id set from the
   * image header by sam_perception_node).  Transform it into the MoveIt
   * planning frame (typically "world" or the robot's base link) so the
   * coordinates are meaningful for arm planning.
   */
  const std::string planning_frame = move_group_->getPlanningFrame();
  try {
    target_out = tf_buffer_->transform(
      res->target_position, planning_frame, tf2::durationFromSec(1.0));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR(this->get_logger(), "TF transform failed: %s", ex.what());
    return false;
  }

  RCLCPP_INFO(this->get_logger(),
    "Target in '%s': (%.3f, %.3f, %.3f) m",
    planning_frame.c_str(),
    target_out.point.x, target_out.point.y, target_out.point.z);
  return true;
}

// ---------------------------------------------------------------------------
// Step 3 — move gripper to pick position (approach then descend)
// ---------------------------------------------------------------------------

bool MultiRateServoNode::move_to_pick_pose(
  const geometry_msgs::msg::PointStamped & target)
{
  auto pick_pose = build_pick_pose(target);

  // Approach pose: same XY position but raised by approach_height_m_ in Z
  auto approach_pose = pick_pose;
  approach_pose.pose.position.z += approach_height_m_;

  moveit::planning_interface::MoveGroupInterface::Plan plan;

  // First move to approach height to avoid sweeping into the object
  move_group_->setPoseTarget(approach_pose);
  if (move_group_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(this->get_logger(), "Planning to approach pose failed");
    return false;
  }
  if (move_group_->execute(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(this->get_logger(), "Executing approach pose failed");
    return false;
  }

  // Descend vertically to the pick height
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

// ---------------------------------------------------------------------------
// Helper — build a downward-facing PoseStamped from a target PointStamped
// ---------------------------------------------------------------------------

geometry_msgs::msg::PoseStamped MultiRateServoNode::build_pick_pose(
  const geometry_msgs::msg::PointStamped & target) const
{
  geometry_msgs::msg::PoseStamped pose;
  pose.header       = target.header;
  pose.pose.position = target.point;

  /*
   * Orientation: gripper pointing straight down.
   * RPY(π, 0, 0) rotates the end-effector 180° around X so the
   * gripper faces the -Z direction (downward in the world frame).
   */
  tf2::Quaternion q;
  q.setRPY(M_PI, 0.0, 0.0);
  q.normalize();
  pose.pose.orientation.x = q.x();
  pose.pose.orientation.y = q.y();
  pose.pose.orientation.z = q.z();
  pose.pose.orientation.w = q.w();

  return pose;
}
