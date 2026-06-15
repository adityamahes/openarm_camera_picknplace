// =============================================================================
// control_node.cpp
// =============================================================================
// Implements ControlNode.  See control_node.hpp for the class overview,
// ownership graph, and threading model.
// =============================================================================

#include "openarm_perception_control/control_node.hpp"

#include <chrono>                       // std::chrono::seconds; chrono_literals (500ms, 30s)
#include <tf2/LinearMath/Quaternion.h>  // tf2::Quaternion — converts RPY angles to a quaternion

using namespace std::chrono_literals;   // lets us write 500ms and 30s as literals below


// =============================================================================
// Constructor
// =============================================================================
// Called once by make_shared<ControlNode>() in main.cpp.
// Sets up everything the node needs EXCEPT MoveGroupInterface (see init()).
ControlNode::ControlNode(const rclcpp::NodeOptions & options)
: Node("control_node", options)   // "control_node" is the ROS 2 node name seen by ros2 node list
{
  // ---------------------------------------------------------------------------
  // Parameters
  // ---------------------------------------------------------------------------
  // Declaring parameters with defaults lets the node run standalone (no yaml).
  // When the launch file loads scan_pose.yaml, those values override these defaults.

  // 7 joint angles in radians that move the arm to an overhead camera position.
  // Order matches joint index in the URDF (j1 … j7).
  this->declare_parameter(
    "scan_joint_values",
    std::vector<double>{0.0, -0.3, 0.0, 1.6, 0.0, 1.57, 0.0});

  // Metres above the pick target the gripper first moves to, before descending.
  this->declare_parameter("approach_height", 0.15);

  // Seconds the OMPL planner is given to find a trajectory per plan() call.
  this->declare_parameter("planning_timeout", 10.0);

  // Read the now-declared parameters into member variables.
  scan_joint_values_  = this->get_parameter("scan_joint_values").as_double_array();
  approach_height_m_  = this->get_parameter("approach_height").as_double();
  planning_timeout_s_ = this->get_parameter("planning_timeout").as_double();

  // ---------------------------------------------------------------------------
  // TF2
  // ---------------------------------------------------------------------------
  // tf_buffer_ is the in-memory store of the entire ROS transform tree.
  // Passing get_clock() ties lookups to the node's time source — this is
  // important in simulation where /clock drives time instead of the wall clock.
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());

  // tf_listener_ subscribes to /tf and /tf_static and feeds every received
  // transform into tf_buffer_.  The constructor takes *tf_buffer_ (a reference,
  // not a copy) so it writes directly into the buffer we own.
  // tf_buffer_ must outlive tf_listener_: both are shared_ptrs owned by this
  // node, and C++ destroys members in reverse declaration order (see the header),
  // so tf_listener_ is destroyed first — safe.
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // ---------------------------------------------------------------------------
  // Callback groups
  // ---------------------------------------------------------------------------
  // Two separate MutuallyExclusive groups are required so that:
  //   - on_pick_prompt can block inside wait_for_target (cbg_prompt_ thread)
  //   - on_pick_target can still fire to deliver the result (cbg_target_ thread)
  // Without two groups, a single-threaded executor would deadlock at the
  // condition variable wait inside wait_for_target.
  cbg_prompt_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  cbg_target_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  // ---------------------------------------------------------------------------
  // /pick_prompt subscription
  // ---------------------------------------------------------------------------
  // Operator or test script publishes a text string here to start a pick.
  // std::bind creates a callable that, when invoked with a SharedPtr<String>,
  // calls this->on_pick_prompt(msg).  std::placeholders::_1 is the first
  // argument slot — filled with the incoming message pointer by the executor.
  rclcpp::SubscriptionOptions prompt_opts;
  prompt_opts.callback_group = cbg_prompt_;   // pin to the prompt thread
  prompt_sub_ = this->create_subscription<std_msgs::msg::String>(
    "/pick_prompt", 10,
    std::bind(&ControlNode::on_pick_prompt, this, std::placeholders::_1),
    prompt_opts);

  // ---------------------------------------------------------------------------
  // /segment_prompt publisher
  // ---------------------------------------------------------------------------
  // Used to forward the text prompt to sam_perception_node so it knows
  // which object to find in the current camera frame.
  segment_pub_ = this->create_publisher<std_msgs::msg::String>("/segment_prompt", 10);

  // ---------------------------------------------------------------------------
  // /pick_target subscription
  // ---------------------------------------------------------------------------
  // sam_perception_node publishes the 3D centroid of the found object here.
  // Bound to cbg_target_ so it runs on a separate thread and can fire while
  // on_pick_prompt is sleeping in wait_for_target.
  rclcpp::SubscriptionOptions target_opts;
  target_opts.callback_group = cbg_target_;   // pin to the target thread
  target_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
    "/pick_target", 10,
    std::bind(&ControlNode::on_pick_target, this, std::placeholders::_1),
    target_opts);
}


// =============================================================================
// init()
// =============================================================================
// Constructs MoveGroupInterface after the node's shared_ptr already exists.
//
// WHY DEFERRED: MoveGroupInterface(shared_from_this(), "right_arm") calls
// shared_from_this() on the Node base class.  shared_from_this() returns a
// shared_ptr to the current object, but it can only work when a shared_ptr
// already manages *this — which is only true after make_shared<ControlNode>()
// returns.  Calling it inside the constructor would throw std::bad_weak_ptr.
void ControlNode::init()
{
  // "right_arm" is the name of the planning group in openarm_right_arm.srdf.
  // MoveGroupInterface connects to the move_group node (launched separately)
  // over ROS 2 actions (/move_action) and services (/query_planner_interface).
  move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
    shared_from_this(), "right_arm");

  // Cap velocity and acceleration to safe fractions of the joint limits.
  // 0.3 = 30 % of max velocity; 0.1 = 10 % of max acceleration.
  move_group_->setPlanningTime(planning_timeout_s_);
  move_group_->setMaxVelocityScalingFactor(0.3);
  move_group_->setMaxAccelerationScalingFactor(0.1);

  RCLCPP_INFO(this->get_logger(), "Initialized. Planning frame: %s",
    move_group_->getPlanningFrame().c_str());
}


// =============================================================================
// on_pick_prompt
// =============================================================================
// Entry point for a complete pick sequence.  Runs on cbg_prompt_.
// Blocks until the sequence finishes (or fails), so only one pick at a time.
void ControlNode::on_pick_prompt(const std_msgs::msg::String::SharedPtr msg)
{
  // msg is a shared_ptr<std_msgs::msg::String>.
  // msg->data is a std::string, e.g. "red tool".
  RCLCPP_INFO(this->get_logger(), "Pick request: '%s'", msg->data.c_str());

  // Step 1: Move arm to the overhead scan pose so the camera sees the workspace.
  if (!move_to_scan_pose()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to reach scan pose");
    return;
  }

  // Brief pause after motion so camera motion-blur and arm vibration settle
  // before we capture the frame for segmentation.
  rclcpp::sleep_for(500ms);

  // Step 2: Publish the text prompt so sam_perception_node knows what to look for.
  // We build a new String message and copy msg->data into it.
  auto prompt_msg = std_msgs::msg::String();
  prompt_msg.data = msg->data;           // std::string copy
  segment_pub_->publish(prompt_msg);

  // Step 3: Block until sam_perception_node publishes the 3D target (30 s timeout).
  // wait_for_target also runs the TF2 transform from camera frame to planning frame.
  geometry_msgs::msg::PointStamped target;
  if (!wait_for_target(target)) {
    RCLCPP_ERROR(this->get_logger(), "Segmentation timed out");
    return;
  }

  // Step 4: Plan and execute the approach → pick two-step motion.
  if (!move_to_pick_pose(target)) {
    RCLCPP_ERROR(this->get_logger(), "Failed to reach pick pose");
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Pick complete");
}


// =============================================================================
// on_pick_target
// =============================================================================
// Fires on cbg_target_ when sam_perception_node publishes on /pick_target.
// Stores the target point and wakes the waiting on_pick_prompt thread.
void ControlNode::on_pick_target(const geometry_msgs::msg::PointStamped::SharedPtr msg)
{
  // lock_guard: acquires target_mutex_ on construction and releases it
  // automatically when the guard goes out of scope (end of this function).
  // Prevents a race between this callback and wait_for_target, which also
  // reads and writes expecting_target_ and pending_target_.
  std::lock_guard<std::mutex> lock(target_mutex_);

  // Only accept a target if on_pick_prompt is actively waiting for one.
  // This prevents stale /pick_target messages — e.g., a message published for
  // a previous pick that was still sitting in the DDS queue — from corrupting
  // the current pick's target.
  if (expecting_target_) {
    // *msg dereferences the SharedPtr to get a PointStamped value, which is
    // then copied into the std::optional<PointStamped>.  After this, the
    // optional has_value() returns true.
    pending_target_ = *msg;

    // Wake the thread sleeping inside wait_for_target.  notify_one wakes
    // exactly one waiter (there is only ever one waiter here, so notify_one
    // is correct and cheaper than notify_all).
    target_cv_.notify_one();
  }
}


// =============================================================================
// move_to_scan_pose
// =============================================================================
// Joint-space motion to the fixed overhead camera position.
bool ControlNode::move_to_scan_pose()
{
  // Set the planning goal to an exact set of 7 joint angles.
  // The order of values in scan_joint_values_ matches the joint order in the URDF.
  move_group_->setJointValueTarget(scan_joint_values_);

  // Plan: ask move_group's OMPL planner for a collision-free joint trajectory.
  // The Plan struct contains the trajectory and metadata (planning time, etc.).
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  if (move_group_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(this->get_logger(), "Planning to scan pose failed");
    return false;
  }

  // Execute: send the trajectory to the ros2_control joint trajectory controller
  // and wait synchronously until the controller reports completion or failure.
  if (move_group_->execute(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(this->get_logger(), "Executing scan pose failed");
    return false;
  }

  RCLCPP_INFO(this->get_logger(), "Reached scan pose");
  return true;
}


// =============================================================================
// wait_for_target
// =============================================================================
// Blocks until on_pick_target delivers a PointStamped, then transforms it
// from the camera frame into the MoveIt planning frame.
bool ControlNode::wait_for_target(geometry_msgs::msg::PointStamped & target_out)
{
  {
    // Short critical section: raise the gate flag and clear any value left
    // over from a previous pick (pending_target_.reset() makes it nullopt again).
    std::lock_guard<std::mutex> lock(target_mutex_);
    expecting_target_ = true;
    pending_target_.reset();    // std::optional::reset() sets the value to nullopt
  }                             // lock released here

  // unique_lock is required by condition_variable::wait_for because the
  // condition variable must temporarily release the mutex while sleeping,
  // then reacquire it before returning.  lock_guard does not support unlock/relock.
  std::unique_lock<std::mutex> lock(target_mutex_);

  // Sleep until either:
  //   a) on_pick_target calls notify_one() and the predicate returns true, OR
  //   b) 30 seconds elapse.
  // The lambda [this]{ return pending_target_.has_value(); } is rechecked on
  // every wake (including spurious wake-ups) to prevent acting on noise.
  bool got = target_cv_.wait_for(lock, std::chrono::seconds(30),
    [this] { return pending_target_.has_value(); });

  // Lower the gate so on_pick_target ignores any late-arriving messages.
  expecting_target_ = false;

  if (!got) {
    RCLCPP_ERROR(this->get_logger(), "Timed out waiting for segmentation result");
    return false;
  }

  // Transform the 3D point from the camera optical frame (stored in
  // pending_target_->header.frame_id, e.g. "camera_color_optical_frame")
  // into the MoveIt planning frame (usually "world" or "base_link").
  //
  // tf_buffer_->transform<PointStamped> looks up the chain of transforms in
  // tf_buffer_ at the timestamp inside pending_target_->header.stamp.
  // The last argument, tf2::durationFromSec(1.0), allows TF2 to wait up to
  // 1 second for a transform that has not yet arrived (handles small clock skew).
  //
  // *pending_target_ dereferences the std::optional to get the PointStamped value.
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


// =============================================================================
// move_to_pick_pose
// =============================================================================
// Two-step Cartesian approach: first to a safe height above the object,
// then straight down to the pick contact point.
bool ControlNode::move_to_pick_pose(const geometry_msgs::msg::PointStamped & target)
{
  // Build the final pick pose (gripper at object XYZ, pointing straight down).
  auto pick_pose = build_pick_pose(target);

  // Approach pose: copy pick_pose, then raise Z by approach_height_m_.
  // The copy-then-modify pattern avoids modifying pick_pose itself.
  auto approach_pose = pick_pose;
  approach_pose.pose.position.z += approach_height_m_;

  // Re-used across both plan/execute steps.
  moveit::planning_interface::MoveGroupInterface::Plan plan;

  // --- Step 1: Move to approach height ---
  // The gripper moves to approach_pose first so it is directly above the
  // object, then descends vertically in step 2.  This avoids sweeping
  // horizontally through the object.
  move_group_->setPoseTarget(approach_pose);
  if (move_group_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(this->get_logger(), "Planning to approach pose failed");
    return false;
  }
  if (move_group_->execute(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(this->get_logger(), "Executing approach pose failed");
    return false;
  }

  // --- Step 2: Descend to the pick contact point ---
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


// =============================================================================
// build_pick_pose
// =============================================================================
// Converts a 3D target point into a full 6-DOF gripper goal pose.
geometry_msgs::msg::PoseStamped ControlNode::build_pick_pose(
  const geometry_msgs::msg::PointStamped & target) const
{
  geometry_msgs::msg::PoseStamped pose;

  // Copy the header (frame_id and timestamp) so MoveIt knows which coordinate
  // frame this pose is expressed in.  This must match the planning frame or
  // MoveIt will fail to interpret the goal.
  pose.header = target.header;

  // The XYZ position is exactly the centroid of the segmented object
  // as computed by sam_perception_node's _mask_to_3d method.
  pose.pose.position = target.point;

  // Orientation: gripper Z-axis pointing straight down.
  // RPY = (π, 0, 0) — rolling 180° flips the gripper Z from "up" to "down".
  // tf2::Quaternion performs the conversion:
  //   setRPY builds a quaternion from intrinsic XYZ Euler angles.
  //   normalize() forces |q| = 1 to eliminate floating-point drift.
  tf2::Quaternion q;
  q.setRPY(M_PI, 0.0, 0.0);
  q.normalize();

  // tf2::Quaternion stores (x, y, z, w) internally; copy each component
  // into the geometry_msgs::msg::Quaternion sub-message.
  pose.pose.orientation.x = q.x();
  pose.pose.orientation.y = q.y();
  pose.pose.orientation.z = q.z();
  pose.pose.orientation.w = q.w();

  return pose;
}
