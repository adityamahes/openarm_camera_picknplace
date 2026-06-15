# OpenArm Perception Pipeline

Text-prompted pick-and-place for the OpenArm robot using a wrist-mounted Intel RealSense camera, GroundingDINO, MobileSAM, and MoveIt 2.

---

## Overview

A user publishes a plain-English description of an object (e.g. `"red tool"`) to the `/pick_prompt` topic. The system locates it on the table, computes its 3D position, and moves the arm to pick it up — no pre-defined object poses required.

```
User:  ros2 topic pub /pick_prompt std_msgs/String "{data: 'red tool'}"
                         │
                         ▼
        control_node  (C++)
        ┌──────────────────────────────────────┐
        │  1. MoveIt → scan pose               │
        │     (arm overhead, camera looking    │
        │      down at the workspace)          │
        │                                      │
        │  2. Publish → /segment_prompt ──────►│──► sam_perception_node  (Python)
        │                                      │◄── /pick_target (3D point, camera frame)
        │  3. TF2 transform → planning frame   │
        │                                      │
        │  4. MoveIt → approach → pick         │
        └──────────────────────────────────────┘
```

---

## Package Layout

```
src/
├── openarm_perception_msgs/          # Custom ROS 2 message/service definitions (if used)
│
├── openarm_sam_perception/           # Python node — AI perception
│   ├── openarm_sam_perception/
│   │   └── sam_perception_node.py    # GroundingDINO + MobileSAM + depth → 3D point
│   ├── setup.py
│   ├── setup.cfg
│   └── package.xml
│
└── openarm_perception_control/       # C++ node — arm orchestration
    ├── include/openarm_perception_control/
    │   └── control_node.hpp          # ControlNode class declaration
    ├── src/
    │   ├── control_node.cpp          # ControlNode method implementations
    │   └── main.cpp                  # Entry point + MultiThreadedExecutor setup
    ├── config/
    │   ├── scan_pose.yaml            # Joint angles for the overhead scan position
    │   ├── kinematics.yaml           # KDL IK solver config for right_arm
    │   ├── joint_limits.yaml         # Per-joint velocity/acceleration limits
    │   ├── moveit_controllers.yaml   # Maps trajectory controllers to planning groups
    │   └── ros2_controllers.yaml     # ros2_control controller definitions
    ├── launch/
    │   └── servo_pipeline.launch.py  # Launches the full pipeline (all nodes)
    ├── test/
    │   └── test_pick_object.py       # Manual integration test script
    ├── CMakeLists.txt
    └── package.xml
```

---

## Node Relationships

```
            [Intel RealSense D-series]
             /camera/color/image_raw              (RGB, 30 Hz)
             /camera/aligned_depth_to_color/image_raw  (depth aligned to RGB)
             /camera/color/camera_info            (intrinsic matrix K, once)
                        │
                        ▼
           ┌────────────────────────────┐
           │   sam_perception_node      │  (Python)
           │                            │
           │  GroundingDINO             │  text prompt → bounding boxes (pixel)
           │    └─► MobileSAM           │  bounding box → pixel mask
           │         └─► depth          │  mask + depth + K → 3D centroid
           │                            │
           │  Subscribes: /segment_prompt          │
           │  Publishes:  /pick_target             │
           └────────────┬───────────────┘
                        │  geometry_msgs/PointStamped
                        │  (3D point in camera optical frame)
                        │
           ┌────────────▼───────────────┐
           │   control_node             │  (C++)
           │                            │
           │  Subscribes: /pick_prompt  │  ← operator sends text here
           │  Publishes:  /segment_prompt│ → triggers perception
           │  Subscribes: /pick_target  │  ← receives 3D result
           │                            │
           │  Uses MoveIt move_group ──►│──► [MoveIt / right_arm]
           │  Uses TF2 for frame xform  │
           └────────────────────────────┘
```

### Topic summary

| Topic | Direction | Type | Purpose |
|-------|-----------|------|---------|
| `/pick_prompt` | operator → control_node | `std_msgs/String` | Start a pick |
| `/segment_prompt` | control_node → sam_perception_node | `std_msgs/String` | Trigger segmentation |
| `/pick_target` | sam_perception_node → control_node | `geometry_msgs/PointStamped` | 3D object location |

---

## Hardware

| Component | Details |
|-----------|---------|
| Robot     | OpenArm v2.0 (7-DOF, right arm only) |
| Camera    | Intel RealSense D-series mounted on the right wrist |
| GPU       | Optional — GroundingDINO and MobileSAM fall back to CPU if no CUDA device is found |

Only the **right arm** is used. It carries both the pinch gripper and the wrist camera.

---

## Code Reference

### `SamPerceptionNode` (Python — `sam_perception_node.py`)

| Method | Role |
|--------|------|
| `__init__()` | Declares parameters, loads models, creates subscriptions and publisher |
| `_load_models()` | Loads GroundingDINO (`load_model`) and MobileSAM (`vit_t`) onto GPU/CPU |
| `_rgb_cb(msg)` | Caches the latest colour frame (overwrites on every new frame) |
| `_depth_cb(msg)` | Caches the latest aligned-depth frame |
| `_info_cb(msg)` | Stores the camera intrinsic matrix `K` once (does not change at runtime) |
| `_prompt_cb(msg)` | Main handler — snapshots the cached frames, runs steps 1–3, publishes result |
| `_run_grounding_dino(bgr, prompt)` | BGR image + text → sorted list of pixel bounding boxes `[x1,y1,x2,y2]` |
| `_run_mobile_sam(bgr, box_xyxy)` | BGR image + box → boolean `H×W` pixel mask |
| `_mask_to_3d(mask, depth, info)` | Mask + depth + intrinsics → `[X, Y, Z]` metres in camera frame |

**`_mask_to_3d` back-projection formula (pinhole camera model):**
```
Camera intrinsic matrix K (from CameraInfo.k, row-major 9 elements):
    K = [ fx   0  cx ]      fx = k[0], cx = k[2]
        [  0  fy  cy ]      fy = k[4], cy = k[5]
        [  0   0   1 ]

For each valid masked pixel at column u, row v with depth d (metres):
    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d

Final 3D point = mean of [X, Y, Z] over all valid masked pixels.
Valid = depth in [0.05 m, 2.5 m] (filters sensor noise and out-of-range readings).
```

---

### `ControlNode` (C++ — `control_node.hpp` / `control_node.cpp`)

| Method | Role |
|--------|------|
| `ControlNode()` | Declares params, creates TF2 buffer/listener, two callback groups, pub/subs |
| `init()` | Creates `MoveGroupInterface("right_arm")` — deferred because it calls `shared_from_this()` |
| `on_pick_prompt(msg)` | `/pick_prompt` handler — drives scan → segment → pick in sequence; blocks on cbg_prompt_ |
| `on_pick_target(msg)` | `/pick_target` handler — stores the point in `pending_target_`, signals `target_cv_` |
| `move_to_scan_pose()` | Joint-space `plan()` + `execute()` to the overhead scan configuration |
| `wait_for_target(out)` | Blocks on `target_cv_` (30 s timeout), then TF2-transforms the point to planning frame |
| `move_to_pick_pose(target)` | Plans/executes approach pose, then descends to the pick contact pose |
| `build_pick_pose(target)` | Converts `PointStamped` to `PoseStamped` with gripper-down orientation (RPY = π, 0, 0) |

**Threading model (MultiThreadedExecutor):**
```
Thread A (cbg_prompt_)                Thread B (cbg_target_)
──────────────────────────────        ──────────────────────────
on_pick_prompt() enters
  move_to_scan_pose()  …waits…
  publish /segment_prompt
  wait_for_target():
    target_cv_.wait_for(30s) ─┐                      sam_perception_node publishes
                               │       on_pick_target() fires
                               │         pending_target_ = *msg
                               │         target_cv_.notify_one()
    woken ◄────────────────────┘
  TF2 transform
  move_to_pick_pose()
```

A `SingleThreadedExecutor` would deadlock here: Thread A would never yield the spin loop, so Thread B's callback would never run.

---

### Entry point (`main.cpp`)

```
rclcpp::init()
  → make_shared<ControlNode>()        constructor: params, TF2, pubs/subs
  → node->init()                      deferred: MoveGroupInterface("right_arm")
  → MultiThreadedExecutor.spin()      dispatch callbacks to threads
  → rclcpp::shutdown()                clean up on Ctrl-C
```

---

## Prerequisites

### ROS 2 packages (Humble)
```bash
sudo apt install \
  ros-humble-moveit \
  ros-humble-realsense2-camera \
  ros-humble-tf2-ros \
  ros-humble-tf2-geometry-msgs \
  ros-humble-cv-bridge
```

### Python packages
```bash
pip install torch torchvision groundingdino-py mobile-sam
```

### Model checkpoints

Place checkpoints in `~/openarm_ws/models/` (the launch file auto-discovers this path):

| Model | File | Source |
|-------|------|--------|
| GroundingDINO config | `GroundingDINO_SwinT_OGC.py` | [GroundingDINO repo](https://github.com/IDEA-Research/GroundingDINO) |
| GroundingDINO weights | `groundingdino_swint_ogc.pth` | Same repo releases |
| MobileSAM weights | `mobile_sam.pt` | [MobileSAM repo](https://github.com/ChaoningZhang/MobileSAM) |

---

## Build

```bash
cd ~/openarm_ws
source /opt/ros/humble/setup.bash

colcon build --packages-select \
  openarm_sam_perception \
  openarm_perception_control

source install/setup.bash
```

---

## Launch

The launch file starts all nodes (robot_state_publisher, ros2_control, move_group, RViz, sam_perception_node, control_node):

```bash
ros2 launch openarm_perception_control servo_pipeline.launch.py
```

Override model paths if they are not in `~/openarm_ws/models/`:

```bash
ros2 launch openarm_perception_control servo_pipeline.launch.py \
  grounding_dino_config:=/path/to/GroundingDINO_SwinT_OGC.py \
  grounding_dino_checkpoint:=/path/to/groundingdino_swint_ogc.pth \
  mobile_sam_checkpoint:=/path/to/mobile_sam.pt
```

---

## Trigger a Pick

```bash
# Publish directly to the pick prompt topic
ros2 topic pub --once /pick_prompt std_msgs/String "{data: 'red tool'}"

# Or use the test script
python3 src/openarm_perception_control/test/test_pick_object.py "red tool"
python3 src/openarm_perception_control/test/test_pick_object.py "blue screwdriver" --timeout 180
```

---

## Scan Pose Calibration

The scan pose (joint angles that position the camera directly above the workspace) is defined in `config/scan_pose.yaml`:

```yaml
scan_joint_values: [0.0, -0.3, 0.0, 1.6, 0.0, 1.57, 0.0]
#                   j1    j2   j3   j4   j5    j6    j7
```

Use `joint_state_publisher_gui` or the RViz joints panel to find the correct values for your physical setup, then update this file before running pick tasks.

### Tuning parameters

| Parameter | Default | Where set | Description |
|-----------|---------|-----------|-------------|
| `scan_joint_values` | `[0,−0.3,0,1.6,0,1.57,0]` | `scan_pose.yaml` | Overhead camera joint configuration (7 values, radians) |
| `approach_height` | `0.15` m | `scan_pose.yaml` | Height above object before descending |
| `planning_timeout` | `10.0` s | `scan_pose.yaml` | MoveIt OMPL time budget per plan call |
| `box_threshold` | `0.35` | launch file | GroundingDINO box confidence cutoff |
| `text_threshold` | `0.25` | launch file | GroundingDINO text-match cutoff |
| `depth_scale` | `0.001` | launch file | RealSense raw depth unit → metres (mm → m) |

---

## Known Issues / Setup Notes

- **transformers >= 5.x**: The launch file's `sam_perception_node` automatically patches `PreTrainedModel.get_head_mask` at startup if it is missing. No manual downgrade needed.
- **Controller timing**: The staggered `TimerAction` delays (5 s / 7 s / 9 s) give `controller_manager` time to come up before spawners run. Increase these on slow machines.
- **Underlay**: Always `source /opt/ros/humble/setup.bash` before `source install/setup.bash`. Running `source install/setup.bash` alone stacks on a stale underlay and causes symbol conflicts.
- **Model paths**: If checkpoint files are not found at startup, `sam_perception_node` will warn but continue. Models load on the first prompt once files become available.
