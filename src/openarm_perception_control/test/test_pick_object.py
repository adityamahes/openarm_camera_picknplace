#!/usr/bin/env python3
"""
test_pick_object.py
===================
Manual integration test for the /pick_object service.

What it tests
-------------
- The /pick_object service is reachable (openarm_perception_control is running).
- The full pipeline completes without error for a given text prompt:
    scan pose → segmentation → arm pick motion.
- The returned final_position is a valid, non-zero 3D point.

Usage
-----
    python3 test_pick_object.py                      # default prompt: "red tool"
    python3 test_pick_object.py "blue screwdriver"
    python3 test_pick_object.py "red tool" --timeout 180

Prerequisites
-------------
All of the following must be running before this test:
    ros2 launch openarm_perception_control servo_pipeline.launch.py ...
    (which starts both openarm_sam_perception and openarm_perception_control)
    + MoveIt move_group
    + RealSense camera driver

Exit codes
----------
    0 — service returned success=True
    1 — service returned success=False, timed out, or was unavailable
"""
import argparse
import sys

import rclpy
from rclpy.node import Node

from openarm_perception_msgs.srv import PickObject


class PickObjectTestClient(Node):
    """
    Minimal ROS 2 node that sends one /pick_object request and reports
    the result.

    Parameters
    ----------
    text_prompt : str
        Natural-language description of the object to pick.
    timeout_s   : float
        Maximum seconds to wait for the service response before giving up.
    """

    def __init__(self, text_prompt: str, timeout_s: float):
        super().__init__('pick_object_test_client')
        self.text_prompt = text_prompt
        self.timeout_s   = timeout_s

        # Create the client; the service name must match what
        # MultiRateServoNode registers in its constructor
        self.client = self.create_client(PickObject, 'pick_object')

    def run(self) -> bool:
        """
        Block until the service call completes (success or failure).

        Returns True if the pick succeeded, False otherwise.
        """
        self.get_logger().info('Waiting for /pick_object service …')

        # Give the node 10 s to come online before declaring it unavailable
        if not self.client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                'Service /pick_object not available after 10 s.\n'
                'Make sure openarm_perception_control is running.')
            return False

        # Build and send the request
        req = PickObject.Request()
        req.text_prompt = self.text_prompt
        self.get_logger().info(f"Sending request: '{self.text_prompt}'")

        future = self.client.async_send_request(req)

        # Poll with spin_once so ROS callbacks (e.g. TF) stay alive while
        # we wait.  A manual deadline is used instead of spin_until_future_complete
        # so we can surface a meaningful timeout message.
        deadline_ns = self.get_clock().now().nanoseconds + int(self.timeout_s * 1e9)
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if future.done():
                break
            if self.get_clock().now().nanoseconds > deadline_ns:
                self.get_logger().error(
                    f'Timed out waiting for response after {self.timeout_s} s')
                return False

        result = future.result()
        if result is None:
            self.get_logger().error('Service call failed — no response object')
            return False

        if result.success:
            # Pretty-print the returned pick position
            p     = result.final_position.point
            frame = result.final_position.header.frame_id
            self.get_logger().info(
                f'\n'
                f'  SUCCESS\n'
                f'  Prompt  : "{self.text_prompt}"\n'
                f'  Position: ({p.x:.4f}, {p.y:.4f}, {p.z:.4f}) m  [{frame}]'
            )
            return True
        else:
            self.get_logger().error(
                f'\n'
                f'  FAILED\n'
                f'  Prompt : "{self.text_prompt}"\n'
                f'  Reason : {result.message}'
            )
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Integration test for the /pick_object service',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'text_prompt',
        nargs='?',
        default='red tool',
        help="Object to pick, e.g. 'red tool' or 'blue screwdriver' (default: 'red tool')",
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=120.0,
        metavar='SECONDS',
        help='Max seconds to wait for the pipeline to complete (default: 120)',
    )
    args = parser.parse_args()

    rclpy.init()
    node    = PickObjectTestClient(args.text_prompt, args.timeout)
    success = node.run()
    node.destroy_node()
    rclpy.shutdown()

    # Non-zero exit code makes this usable in shell scripts / CI
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
