#pragma once

// =============================================================================
// control_node.hpp
// =============================================================================
// Declares ControlNode — the C++ ROS 2 node that orchestrates the full
// text-prompted pick-and-place pipeline for the OpenArm right arm.
//
// High-level sequence (triggered by a message on /pick_prompt):
//   1. Move the arm to the scan pose — a fixed overhead joint configuration
//      that positions the wrist camera directly above the workspace.
//   2. Forward the text prompt to the perception node via /segment_prompt.
//   3. Block (with a 30-second timeout) until the perception node responds
//      with a 3D point on /pick_target.
//   4. Transform that point from the camera frame into the MoveIt planning frame
//      using the TF2 transform tree.
//   5. Execute a two-step approach-then-descend Cartesian motion to the object.
//
// Threading model
// ---------------
// ControlNode must run under a MultiThreadedExecutor (see main.cpp).
// Two MutuallyExclusive callback groups are created:
//
//   cbg_prompt_  — handles /pick_prompt callbacks.
//                  Runs on Thread A and blocks for the full pick duration.
//
//   cbg_target_  — handles /pick_target callbacks.
//                  Runs on Thread B, independent of cbg_prompt_.
//
// This separation is critical: on_pick_prompt blocks on a std::condition_variable
// while waiting for the perception result.  on_pick_target fires on a different
// executor thread and signals that condition variable to unblock it.
// A single-threaded executor would deadlock here.
// =============================================================================

#include <condition_variable>   // std::condition_variable — blocks/signals across threads
#include <mutex>                // std::mutex, std::lock_guard, std::unique_lock
#include <optional>             // std::optional — nullable value stored inline (no heap alloc)
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
// MoveGroupInterface: high-level API to MoveIt's move_group node.
// Used to set goals (joint-space or Cartesian), call the planner, and execute trajectories.
#include <moveit/move_group_interface/move_group_interface.h>
// PointStamped: a 3D point (x, y, z) tagged with a coordinate frame and timestamp.
#include <geometry_msgs/msg/point_stamped.hpp>
// PoseStamped: a 6-DOF pose (position + quaternion orientation) with frame + timestamp.
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/string.hpp>
// tf2_ros::Buffer: thread-safe store of the live TF2 transform tree.
#include <tf2_ros/buffer.h>
// tf2_ros::TransformListener: subscribes to /tf and /tf_static; fills tf_buffer_.
#include <tf2_ros/transform_listener.h>
// Provides tf2::doTransform overloads for geometry_msgs types (PointStamped, etc.).
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>


// ControlNode
// -----------
// Single ROS 2 node that drives the OpenArm right arm through a
// perception-guided pick sequence.
//
// Ownership graph (all members are shared_ptrs — reference-counted lifetimes):
//
//   ControlNode  (shared_ptr held by the executor in main.cpp)
//     ├── move_group_   → MoveGroupInterface("right_arm")
//     ├── tf_buffer_    → tf2_ros::Buffer   ← tf_listener_ populates this
//     ├── tf_listener_  → tf2_ros::TransformListener
//     ├── cbg_prompt_   → rclcpp::CallbackGroup (MutuallyExclusive)
//     ├── cbg_target_   → rclcpp::CallbackGroup (MutuallyExclusive)
//     ├── prompt_sub_   → Subscription<String>       (/pick_prompt)
//     ├── segment_pub_  → Publisher<String>           (/segment_prompt)
//     └── target_sub_   → Subscription<PointStamped>  (/pick_target)
class ControlNode : public rclcpp::Node
{
public:
  // Constructor
  // -----------
  // Declares ROS 2 parameters, creates the TF2 buffer/listener, callback
  // groups, and all subscriptions/publishers.
  //
  // MoveGroupInterface is NOT constructed here because its constructor calls
  // shared_from_this() — which requires a shared_ptr<Node> to already manage
  // this object.  That is only guaranteed true after make_shared<ControlNode>()
  // returns to the caller.  Call init() immediately afterwards (see main.cpp).
  explicit ControlNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  // init()
  // ------
  // Creates the MoveGroupInterface for the "right_arm" planning group and
  // configures velocity/acceleration scaling and the planning time budget.
  // Must be called once, immediately after make_shared<ControlNode>().
  void init();

private:
  // =========================================================================
  // Callbacks
  // =========================================================================

  // on_pick_prompt
  // --------------
  // Fires on cbg_prompt_ when a string arrives on /pick_prompt.
  // msg->data holds the plain-English object name (e.g. "red tool").
  //
  // Owns the full pick sequence:
  //   move_to_scan_pose → publish /segment_prompt → wait_for_target → move_to_pick_pose
  //
  // Blocks the cbg_prompt_ thread for the entire pick duration, so at most
  // one pick is active at a time.
  void on_pick_prompt(const std_msgs::msg::String::SharedPtr msg);

  // on_pick_target
  // --------------
  // Fires on cbg_target_ when sam_perception_node publishes on /pick_target.
  // msg is a PointStamped expressed in the camera optical frame.
  //
  // Stores *msg into pending_target_ and calls target_cv_.notify_one() to wake
  // the wait_for_target() call that is sleeping inside on_pick_prompt.
  // Ignores messages that arrive outside the expected window (expecting_target_ == false).
  void on_pick_target(const geometry_msgs::msg::PointStamped::SharedPtr msg);

  // =========================================================================
  // Motion helpers
  // =========================================================================

  // move_to_scan_pose
  // -----------------
  // Joint-space motion to the overhead camera position defined by scan_joint_values_.
  // Returns true on success, false if MoveIt planning or execution fails.
  bool move_to_scan_pose();

  // wait_for_target
  // ---------------
  // Sets expecting_target_ = true, then blocks on target_cv_ for up to 30 s.
  // When on_pick_target fires and fills pending_target_, this function wakes,
  // TF2-transforms the point from the camera frame into the MoveIt planning
  // frame, and writes the result into target_out.
  // Returns false on timeout or if TF2 cannot find the required transform.
  bool wait_for_target(geometry_msgs::msg::PointStamped & target_out);

  // move_to_pick_pose
  // -----------------
  // Two-step Cartesian approach:
  //   Step 1 — approach pose: same XY as target, Z raised by approach_height_m_.
  //   Step 2 — pick pose:     same XY, same Z as target (the object centroid).
  // Returns false if any plan() or execute() call fails.
  bool move_to_pick_pose(const geometry_msgs::msg::PointStamped & target);

  // build_pick_pose
  // ---------------
  // Converts a PointStamped into a PoseStamped at the same XYZ position.
  // Orientation is fixed: roll=π, pitch=0, yaw=0 — rotates the gripper Z-axis
  // from "pointing up" to "pointing straight down at the table surface".
  // The header (frame_id + stamp) is copied from target so MoveIt knows the frame.
  geometry_msgs::msg::PoseStamped build_pick_pose(
    const geometry_msgs::msg::PointStamped & target) const;

  // =========================================================================
  // MoveIt
  // =========================================================================

  // move_group_ is a shared_ptr to MoveGroupInterface("right_arm").
  // "right_arm" is the planning group name defined in openarm_right_arm.srdf.
  //
  // shared_ptr: ControlNode retains shared ownership.  The MoveGroupInterface
  // is destroyed when the last shared_ptr to it drops — here, when the node
  // itself is destroyed (executor releases its shared_ptr to ControlNode).
  //
  // Constructed in init() rather than the constructor because
  // MoveGroupInterface internally calls shared_from_this(), which requires
  // a shared_ptr to already manage *this.
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  // =========================================================================
  // TF2
  // =========================================================================

  // tf_buffer_ holds the live, thread-safe TF2 transform tree.
  // Constructed with get_clock() so that transform lookups and timeout
  // calculations use the node's time source (wall clock on real hardware,
  // /clock in simulation).
  //
  // IMPORTANT: tf_buffer_ must outlive tf_listener_, because tf_listener_
  // holds a raw reference to tf_buffer_ (*tf_buffer_) in its constructor.
  // Declaring tf_buffer_ before tf_listener_ in this header ensures it is
  // constructed first and destroyed last (C++ constructs in declaration order,
  // destroys in reverse order).
  std::shared_ptr<tf2_ros::Buffer>            tf_buffer_;

  // tf_listener_ subscribes to /tf and /tf_static and writes incoming
  // transforms into tf_buffer_.  Keeping it alive (via shared_ptr) is required
  // to keep the subscription active; destroying it stops the updates.
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // =========================================================================
  // Callback groups
  // =========================================================================

  // cbg_prompt_ — MutuallyExclusive group for /pick_prompt.
  // "MutuallyExclusive" means the executor runs at most one callback from this
  // group at a time, so concurrent calls to on_pick_prompt cannot overlap.
  rclcpp::CallbackGroup::SharedPtr cbg_prompt_;

  // cbg_target_ — MutuallyExclusive group for /pick_target.
  // Assigned to a separate executor thread so on_pick_target can fire even
  // while on_pick_prompt is blocked inside wait_for_target().
  rclcpp::CallbackGroup::SharedPtr cbg_target_;

  // =========================================================================
  // Publishers and Subscribers
  // =========================================================================

  // prompt_sub_ — receives text pick requests from the operator or test script.
  // Topic: /pick_prompt  |  Type: std_msgs/String  |  Queue depth: 10
  // Bound to cbg_prompt_.
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr prompt_sub_;

  // segment_pub_ — forwards the text prompt to sam_perception_node so it
  // knows which object to locate in the current camera frame.
  // Topic: /segment_prompt  |  Type: std_msgs/String  |  Queue depth: 10
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr segment_pub_;

  // target_sub_ — receives the 3D centroid of the segmented object.
  // Topic: /pick_target  |  Type: geometry_msgs/PointStamped  |  Queue depth: 10
  // Bound to cbg_target_.
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr target_sub_;

  // =========================================================================
  // Cross-thread target handoff (mutex + condition variable pattern)
  // =========================================================================
  //
  // Problem: on_pick_prompt (Thread A) must wait for a result that arrives on
  //          on_pick_target (Thread B).
  //
  // Solution:
  //   target_mutex_     — guards pending_target_ and expecting_target_ so the
  //                        two threads do not race on those variables.
  //   target_cv_        — condition variable.  Thread A calls wait_for() (releases
  //                        the mutex and sleeps).  Thread B calls notify_one()
  //                        (wakes Thread A).
  //   pending_target_   — carries the PointStamped from Thread B to Thread A.
  //   expecting_target_ — gate flag: on_pick_target ignores messages unless
  //                        on_pick_prompt has raised this flag, preventing stale
  //                        messages from a prior pick from being used.

  std::mutex              target_mutex_;
  std::condition_variable target_cv_;

  // pending_target_ uses std::optional so it is clearly "empty" (std::nullopt)
  // before a target arrives.  The wait predicate [this]{ return pending_target_.has_value(); }
  // guards against spurious condition-variable wake-ups.
  std::optional<geometry_msgs::msg::PointStamped> pending_target_;

  // expecting_target_ is true only during the window between publishing
  // /segment_prompt and receiving the /pick_target response.
  bool expecting_target_{false};

  // =========================================================================
  // Parameters
  // =========================================================================

  // scan_joint_values_: 7 joint angles in radians defining the overhead scan pose.
  // Populated from the scan_pose.yaml config file (loaded in the launch file).
  // Index order matches the joint order in the URDF.
  std::vector<double> scan_joint_values_;

  // approach_height_m_: metres to rise above the pick target before descending.
  // Prevents the gripper from knocking the object over on the way in.
  double approach_height_m_;

  // planning_timeout_s_: maximum seconds OMPL is allowed per plan() call.
  // Increase if planning fails in cluttered scenes.
  double planning_timeout_s_;
};
