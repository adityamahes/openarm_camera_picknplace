#pragma once

/*
 * multi_rate_servo_node.hpp
 * =========================
 * Class declaration for the C++ arm-orchestration node.
 *
 * This node owns the complete pick-and-place pipeline:
 *   1. Move the right arm to a fixed "scan" pose so the wrist camera
 *      looks straight down at the workspace table.
 *   2. Call the /segment_object service (provided by openarm_sam_perception)
 *      with a text description of the target object.
 *   3. Transform the returned 3D point from the camera frame into the
 *      MoveIt planning frame via TF2.
 *   4. Move the gripper to an approach position above the target, then
 *      descend to the pick position.
 *
 * Concurrency design
 * ------------------
 * The node runs inside a MultiThreadedExecutor so that:
 *   - The /pick_object service handler (cbg_pick_service_) can block for the
 *     full duration of the pipeline (arm motion + segmentation + arm motion)
 *     without starving other ROS callbacks.
 *   - The /segment_object response callback (cbg_seg_client_) can be delivered
 *     by another executor thread while the pick handler awaits it.
 *
 * The segmentation call uses async_send_request + std::promise instead of
 * spin_until_future_complete because spin_until_future_complete cannot be
 * called from inside a callback of a node that is already being spun.
 */

#include <future>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <openarm_perception_msgs/srv/segment_object.hpp>
#include <openarm_perception_msgs/srv/pick_object.hpp>

class MultiRateServoNode : public rclcpp::Node
{
public:
  /*
   * Constructor — declares ROS parameters, creates the TF listener, the
   * segmentation service client, and the pick_object service server.
   *
   * Does NOT create MoveGroupInterface here because shared_from_this() is
   * unavailable inside a constructor.  Call init() after make_shared().
   */
  explicit MultiRateServoNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  /*
   * init() — must be called once after the node is wrapped in a shared_ptr.
   * Creates MoveGroupInterface for the "right_arm" planning group and
   * applies velocity/acceleration scaling limits.
   */
  void init();

private:
  /*
   * handle_pick_object()
   * Service handler for /pick_object.
   * Runs the full pipeline: scan → segment → pick.
   * Blocks until completion (on its own callback group, so other callbacks
   * remain alive on the MultiThreadedExecutor).
   */
  void handle_pick_object(
    std::shared_ptr<openarm_perception_msgs::srv::PickObject::Request> request,
    std::shared_ptr<openarm_perception_msgs::srv::PickObject::Response> response);

  /*
   * move_to_scan_pose()
   * Calls MoveIt plan + execute to reach the configured scan joint values.
   * Returns true on success.
   */
  bool move_to_scan_pose();

  /*
   * call_segmentation()
   * Sends a /segment_object request with the given text prompt.
   * Transforms the camera-frame result into the MoveIt planning frame.
   * Uses std::promise so the executor can deliver the response callback
   * concurrently while this function blocks on std::future::get().
   * Returns true on success and fills target_out.
   */
  bool call_segmentation(
    const std::string & text_prompt,
    geometry_msgs::msg::PointStamped & target_out);

  /*
   * move_to_pick_pose()
   * Moves the gripper to a position above the target (approach), then
   * descends to the target height.  Returns true on success.
   */
  bool move_to_pick_pose(const geometry_msgs::msg::PointStamped & target);

  /*
   * build_pick_pose()
   * Constructs a PoseStamped from a PointStamped target.
   * Orientation is set to gripper-pointing-down (RPY = π, 0, 0).
   */
  geometry_msgs::msg::PoseStamped build_pick_pose(
    const geometry_msgs::msg::PointStamped & target) const;

  // MoveIt interface for the right_arm planning group
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  // TF2 infrastructure for camera-frame → planning-frame transforms
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  /*
   * Two separate callback groups allow the MultiThreadedExecutor to run
   * the pick_object handler and the segmentation response callback
   * simultaneously on different threads.
   */
  rclcpp::CallbackGroup::SharedPtr cbg_pick_service_;  // for /pick_object server
  rclcpp::CallbackGroup::SharedPtr cbg_seg_client_;    // for /segment_object client

  rclcpp::Client<openarm_perception_msgs::srv::SegmentObject>::SharedPtr seg_client_;
  rclcpp::Service<openarm_perception_msgs::srv::PickObject>::SharedPtr pick_service_;

  // Loaded from scan_pose.yaml / ROS parameters
  std::vector<double> scan_joint_values_;  // 7 joint angles (radians) for the scan pose
  double approach_height_m_;               // metres above target before descending
  double planning_timeout_s_;              // MoveIt planning time budget
};
