// =============================================================================
// main.cpp
// =============================================================================
// Entry point for the control_node executable.
//
// Execution flow:
//   1. rclcpp::init        — parse ROS 2 arguments; connect to the DDS middleware
//   2. make_shared<>       — heap-allocate ControlNode; constructor runs (pubs/subs/params)
//   3. node->init()        — create MoveGroupInterface (requires shared_ptr to exist first)
//   4. executor.spin()     — dispatch callbacks to threads until Ctrl-C / SIGINT
//   5. rclcpp::shutdown()  — tear down ROS context and middleware connections
//
// Why MultiThreadedExecutor?
// --------------------------
// ControlNode has two callback groups on separate threads:
//
//   cbg_prompt_ (Thread A) — on_pick_prompt blocks for the full pick duration
//                            (planning + execution can take many seconds).
//
//   cbg_target_ (Thread B) — on_pick_target must fire WHILE Thread A is blocked,
//                            to deliver the 3D pick target that unblocks it.
//
// A SingleThreadedExecutor runs callbacks serially — Thread A would never yield,
// Thread B's callback would never run, and the node would deadlock.
// MultiThreadedExecutor assigns each callback group its own thread, so both
// can run concurrently.
// =============================================================================

#include <rclcpp/rclcpp.hpp>
#include "openarm_perception_control/control_node.hpp"

int main(int argc, char ** argv)
{
  // Parse --ros-args (remappings, parameters, node name overrides) and
  // initialise the ROS 2 / DDS communication layer for this process.
  rclcpp::init(argc, argv);

  // Allocate ControlNode on the heap, managed by a shared_ptr<ControlNode>.
  // shared_ptr is mandatory because:
  //   a) ControlNode inherits rclcpp::Node which uses shared_from_this() internally.
  //   b) MoveGroupInterface (created in init()) calls shared_from_this() explicitly.
  //   c) The executor holds a weak_ptr to the node and requires shared ownership.
  // After this line, the constructor has run: parameters declared, TF2 created,
  // callback groups and pub/subs registered — but MoveGroupInterface not yet created.
  auto node = std::make_shared<ControlNode>();

  // Create MoveGroupInterface now that a shared_ptr<ControlNode> exists.
  // Must be called before spin so the node can respond to /pick_prompt messages.
  node->init();

  // MultiThreadedExecutor: spawns a thread pool that delivers callbacks concurrently
  // across different callback groups (see class header for the threading model).
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);   // register the node; the executor now holds a reference

  // spin() blocks here, pulling incoming messages off the DDS queues and
  // dispatching them to the appropriate callback on the appropriate thread.
  // Returns when rclcpp::shutdown() is called — typically on SIGINT (Ctrl-C).
  executor.spin();

  // Release all ROS 2 resources: subscriptions, publishers, services, and the
  // DDS participant.  Safe to call even if init() was not reached.
  rclcpp::shutdown();
  return 0;
}
