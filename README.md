# OpenArm Perception Pipeline

Text-prompted pick-and-place for the OpenArm robot using a wrist-mounted Intel RealSense camera, GroundingDINO, MobileSAM, and MoveIt 2.

---

## Overview

A user types a plain-English description of an object (e.g. `"red tool"`). The system locates it on the table, computes its 3D position, and moves the arm to pick it up — no pre-defined object poses required.

```
User:  ros2 service call /pick_object "red tool"
                         │
                         ▼
        openarm_perception_control  (C++)
        ┌──────────────────────────────────┐
        │  1. MoveIt → scan pose           │
        │     (arm overhead, camera down)  │
        │                                  │
        │  2. Call /segment_object ───────►│──► openarm_sam_perception  (Python)
        │                                  │◄── 3D point in camera frame
        │  3. TF2 transform to world frame │
        │                                  │
        │  4. MoveIt → approach → pick     │
        └──────────────────────────────────┘
```

---

## Package Layout

```
src/
├── openarm_perception_msgs/          # Custom ROS 2 service definitions
│   ├── srv/
│   │   ├── SegmentObject.srv         # text_prompt → 3D point + bounding box
│   │   └── PickObject.srv            # text_prompt → success + final position
│   ├── CMakeLists.txt
│   └── package.xml
│
├── openarm_sam_perception/           # Python node — AI perception
│   ├── openarm_sam_perception/
│   │   └── sam_perception_node.py    # Node implementation
│   ├── setup.py
│   ├── setup.cfg
│   └── package.xml
│
└── openarm_perception_control/       # C++ node — arm orchestration
    ├── include/openarm_perception_control/
    │   └── multi_rate_servo_node.hpp # Class declaration
    ├── src/
    │   ├── multi_rate_servo_node.cpp # Class implementation
    │   └── main.cpp                  # Entry point + executor setup
    ├── config/
    │   └── scan_pose.yaml            # Tunable joint values for scan pose
    ├── launch/
    │   └── servo_pipeline.launch.py  # Launches both nodes
    ├── test/
    │   └── test_pick_object.py       # Manual integration test
    ├── CMakeLists.txt
    └── package.xml
```

---

## Node Relationships

```
            [Intel RealSense D-series]
             /camera/color/image_raw
             /camera/aligned_depth_to_color/image_raw
             /camera/color/camera_info
                        │
                        ▼
           ┌────────────────────────┐
           │   sam_perception_node  │  (Python)
           │                        │
           │  GroundingDINO         │  text → bounding box
           │    └─► MobileSAM       │  box  → pixel mask
           │         └─► depth      │  mask → 3D centroid
           │                        │
           │  Provides service:     │
           │  /segment_object       │
           └────────────┬───────────┘
                        │  SegmentObject.srv
                        │  (text_prompt → PointStamped in camera frame)
                        │
           ┌────────────▼───────────┐
           │ multi_rate_servo_node  │  (C++)
           │                        │
           │  Consumes:             │
           │  /segment_object       │
           │                        │
           │  Uses:                 │
           │  MoveIt move_group ───►│──► [MoveIt / right_arm]
           │  TF2 tree              │
           │                        │
           │  Provides service:     │
           │  /pick_object          │
           └────────────┬───────────┘
                        │  PickObject.srv
                        │  (text_prompt → success + final_position)
                        │
           ┌────────────▼───────────┐
           │  test_pick_object.py   │  (test client / any external caller)
           └────────────────────────┘
```

---

## Hardware

| Component | Details |
|-----------|---------|
| Robot     | OpenArm (7-DOF, right arm only) |
| Camera    | Intel RealSense mounted on the right wrist |
| GPU       | Optional — GroundingDINO and MobileSAM run on CPU if no CUDA device is found |

Only the **right arm** is used. The right arm carries both the gripper and the wrist camera.

---

## Class & Function Reference

### `SamPerceptionNode` (Python — `sam_perception_node.py`)

| Method | Role |
|--------|------|
| `__init__()` | Declares parameters, loads models, creates subscriptions and service server |
| `_load_models()` | Loads GroundingDINO and MobileSAM (`vit_t`) onto GPU/CPU |
| `_rgb_cb()` | Caches the latest colour frame |
| `_depth_cb()` | Caches the latest aligned-depth frame |
| `_info_cb()` | Stores camera intrinsics once (they don't change) |
| `_handle_segment()` | Service handler — orchestrates steps 1–3 below |
| `_run_grounding_dino()` | Text prompt → sorted list of pixel bounding boxes |
| `_run_mobile_sam()` | Bounding box → boolean pixel mask |
| `_mask_to_3d()` | Mask + depth + intrinsics → `[X, Y, Z]` in camera frame |

**`_mask_to_3d` back-projection formula:**
```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth_metres          (mean over all valid masked pixels)
```

---

### `MultiRateServoNode` (C++ — `multi_rate_servo_node.hpp / .cpp`)

| Method | Role |
|--------|------|
| `MultiRateServoNode()` | Declares params, creates TF2 listener, service client, service server, and two callback groups |
| `init()` | Creates `MoveGroupInterface("right_arm")` — must be called after `make_shared()` |
| `handle_pick_object()` | `/pick_object` handler — runs scan → segment → pick in sequence |
| `move_to_scan_pose()` | MoveIt `plan()` + `execute()` to the configured overhead joint values |
| `call_segmentation()` | Calls `/segment_object` via `async_send_request` + `std::promise`, then TF2-transforms the result |
| `move_to_pick_pose()` | Plans + executes approach pose, then descends to pick pose |
| `build_pick_pose()` | Builds a `PoseStamped` from a `PointStamped` with gripper-down orientation (RPY = π, 0, 0) |

---

### `PickObjectTestClient` (Python — `test/test_pick_object.py`)

| Method | Role |
|--------|------|
| `__init__()` | Creates `/pick_object` client |
| `run()` | Waits for service, sends request, polls with `spin_once`, prints result |

---

## Why `std::promise` Instead of `spin_until_future_complete`

`handle_pick_object` is a ROS 2 service callback. Calling `spin_until_future_complete` from inside a callback causes a **deadlock** because it tries to spin the node a second time.

The fix: `async_send_request` paired with a `std::promise`. The `MultiThreadedExecutor` delivers the `/segment_object` response on a **separate callback group** (`cbg_seg_client_`), fulfilling the promise. The pick handler blocks on `std::future::wait_for()` without touching the executor.

```
Thread A (cbg_pick_service_)          Thread B (cbg_seg_client_)
────────────────────────────          ──────────────────────────
handle_pick_object() running
  async_send_request(req, lambda) ──► request sent
  future.wait_for(30s) …              … executor dispatches response
                                      lambda fires → promise.set_value()
  future is ready ◄───────────────────────────────────────────
  result = future.get()
```

---

## Service Definitions

### `SegmentObject.srv`
```
string text_prompt
---
bool success
string message
geometry_msgs/PointStamped target_position
float32[] bounding_box
```

### `PickObject.srv`
```
string text_prompt
---
bool success
string message
geometry_msgs/PointStamped final_position
```

---

## Prerequisites

### ROS 2 packages
```bash
sudo apt install ros-humble-moveit ros-humble-realsense2-camera \
     ros-humble-tf2-ros ros-humble-tf2-geometry-msgs ros-humble-cv-bridge
```

### Python packages
```bash
pip install torch torchvision groundingdino-py mobile-sam
```

### Model checkpoints

Download and note the paths — pass them to the launch file:

| Model | File | Source |
|-------|------|--------|
| GroundingDINO config | `GroundingDINO_SwinT_OGC.py` | [GroundingDINO repo](https://github.com/IDEA-Research/GroundingDINO) |
| GroundingDINO weights | `groundingdino_swint_ogc.pth` | Same repo |
| MobileSAM weights | `mobile_sam.pt` | [MobileSAM repo](https://github.com/ChaoningZhang/MobileSAM) |

---

## Build

```bash
cd ~/openarm_ws
source /opt/ros/humble/setup.bash

colcon build --packages-select \
  openarm_perception_msgs \
  openarm_sam_perception \
  openarm_perception_control

source install/setup.bash
```

---

## Launch

MoveIt `move_group` must already be running before launching this pipeline.

```bash
ros2 launch openarm_perception_control servo_pipeline.launch.py \
  grounding_dino_config:=/path/to/GroundingDINO_SwinT_OGC.py \
  grounding_dino_checkpoint:=/path/to/groundingdino_swint_ogc.pth \
  mobile_sam_checkpoint:=/path/to/mobile_sam.pt
```

---

## Trigger a Pick

```bash
# Pick the first matching object found
ros2 service call /pick_object openarm_perception_msgs/srv/PickObject \
  "{text_prompt: 'red tool'}"

# Or use the test script
python3 src/openarm_perception_control/test/test_pick_object.py "red tool"
python3 src/openarm_perception_control/test/test_pick_object.py "blue screwdriver" --timeout 180
```

---

## Scan Pose Calibration

The scan pose (joint angles that position the camera directly above the workspace) is in `openarm_perception_control/config/scan_pose.yaml`.

```yaml
scan_joint_values: [0.0, -0.3, 0.0, 1.6, 0.0, 1.57, 0.0]
#                   j1    j2   j3   j4   j5    j6    j7
```

Use `joint_state_publisher_gui` or RViz to find the correct values for your physical setup, then update this file before running pick tasks.

Additional tuning parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `approach_height` | `0.15` m | Height above target before descending |
| `planning_timeout` | `10.0` s | MoveIt planning time budget |
| `box_threshold` | `0.35` | GroundingDINO box confidence threshold |
| `text_threshold` | `0.25` | GroundingDINO text match threshold |
| `depth_scale` | `0.001` | RealSense raw depth → metres (mm → m) |
