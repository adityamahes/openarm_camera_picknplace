# OpenArm Perception Pipeline

Text-prompted pick-and-place for the OpenArm robot using a wrist-mounted Intel RealSense camera, GroundingDINO, MobileSAM, and MoveIt 2.

---

## How It Works

You type a plain-English description of an object (e.g. `"red cup"`). The system finds it on the table, computes its 3D position, and moves the arm to pick it up — no pre-defined object poses required.

```
You type:  ros2 topic pub /pick_prompt std_msgs/String "{data: 'red cup'}"
                         │
                         ▼
        control_node  (C++)
        ┌──────────────────────────────────────┐
        │  1. MoveIt → scan pose               │
        │     (arm moves overhead, camera      │
        │      looking down at the workspace)  │
        │                                      │
        │  2. Publish → /segment_prompt ──────►│──► sam_perception_node  (Python)
        │                                      │◄── /pick_target (3D point)
        │  3. TF2 transform → planning frame   │
        │                                      │
        │  4. MoveIt → approach → pick         │
        └──────────────────────────────────────┘
```

---

## Quick Start

### Step 1 — Build (only the pipeline packages)

```bash
# Always source the ROS 2 underlay FIRST, before sourcing the workspace
source /opt/ros/humble/setup.bash

cd ~/openarm_ws

colcon build --packages-select \
  openarm_sam_perception \
  openarm_perception_control
```

### Step 2 — Source the workspace

```bash
# Do this in every terminal you open for this project
source /opt/ros/humble/setup.bash
source ~/openarm_ws/install/setup.bash
```

> **Note:** Always run both `source` lines in this order. Running only
> `source install/setup.bash` stacks on a stale underlay and causes symbol conflicts.

### Step 3 — Launch the full pipeline

```bash
ros2 launch openarm_perception_control servo_pipeline.launch.py
```

Wait for output like `move_group ready` and `sam_perception_node: models loaded` before sending a prompt. Startup takes about 10–15 seconds.

### Step 4 — Send a pick prompt

Open a **new terminal**, source again, then publish your text prompt:

```bash
source /opt/ros/humble/setup.bash
source ~/openarm_ws/install/setup.bash

# Replace "red cup" with whatever object you want to pick
ros2 topic pub --once /pick_prompt std_msgs/String "{data: 'red cup'}"
```

Other example prompts:
```bash
ros2 topic pub --once /pick_prompt std_msgs/String "{data: 'blue screwdriver'}"
ros2 topic pub --once /pick_prompt std_msgs/String "{data: 'yellow bottle'}"
ros2 topic pub --once /pick_prompt std_msgs/String "{data: 'red tool'}"
```

---

## Testing Individual Nodes

You can run each node in isolation to debug without the full pipeline.

### Test only the perception node (camera + AI)

This lets you verify that GroundingDINO and MobileSAM detect your object correctly, without needing the robot arm or MoveIt.

**Terminal 1 — start a RealSense camera node:**
```bash
source /opt/ros/humble/setup.bash
source ~/openarm_ws/install/setup.bash

ros2 launch realsense2_camera rs_launch.py \
  align_depth.enable:=true
```

**Terminal 2 — run the perception node alone:**
```bash
source /opt/ros/humble/setup.bash
source ~/openarm_ws/install/setup.bash

ros2 run openarm_sam_perception sam_perception_node \
  --ros-args \
  -p grounding_dino_config:=~/openarm_ws/models/GroundingDINO_SwinT_OGC.py \
  -p grounding_dino_checkpoint:=~/openarm_ws/models/groundingdino_swint_ogc.pth \
  -p mobile_sam_checkpoint:=~/openarm_ws/models/mobile_sam.pt
```

**Terminal 3 — send a test prompt and check the result:**
```bash
source /opt/ros/humble/setup.bash
source ~/openarm_ws/install/setup.bash

# Trigger segmentation with a text prompt
ros2 topic pub --once /segment_prompt std_msgs/String "{data: 'red cup'}"

# Watch the 3D point that comes back
ros2 topic echo /pick_target
```

If `/pick_target` publishes a point with non-zero Z, the perception node is working.

---

### Test only the control node (arm motion, no camera)

The control node needs MoveIt running but does not need the RealSense camera. Useful for testing arm movements and scan pose calibration.

**Terminal 1 — launch robot + MoveIt only:**
```bash
source /opt/ros/humble/setup.bash
source ~/openarm_ws/install/setup.bash

ros2 launch openarm_perception_control servo_pipeline.launch.py
```

Wait for `move_group ready`.

**Terminal 2 — manually publish a fake 3D target to `/pick_target`:**
```bash
source /opt/ros/humble/setup.bash
source ~/openarm_ws/install/setup.bash

# Publish a fake object location (x=0.3m, y=0.0m, z=0.05m in camera frame)
ros2 topic pub --once /pick_target geometry_msgs/PointStamped \
  "{header: {frame_id: 'camera_color_optical_frame'}, point: {x: 0.3, y: 0.0, z: 0.5}}"
```

Then in a third terminal, trigger the pick sequence:
```bash
source /opt/ros/humble/setup.bash
source ~/openarm_ws/install/setup.bash

ros2 topic pub --once /pick_prompt std_msgs/String "{data: 'test object'}"
```

---

## Model Checkpoints

Download and place the following files in `~/openarm_ws/models/`:

| File | Source |
|------|--------|
| `GroundingDINO_SwinT_OGC.py` | [GroundingDINO repo](https://github.com/IDEA-Research/GroundingDINO) — `groundingdino/config/` |
| `groundingdino_swint_ogc.pth` | Same repo, releases page |
| `mobile_sam.pt` | [MobileSAM repo](https://github.com/ChaoningZhang/MobileSAM) |

```bash
mkdir -p ~/openarm_ws/models
# place the three files above here
ls ~/openarm_ws/models/
# GroundingDINO_SwinT_OGC.py  groundingdino_swint_ogc.pth  mobile_sam.pt
```

If your checkpoints are in a different location, override the paths at launch time:

```bash
ros2 launch openarm_perception_control servo_pipeline.launch.py \
  grounding_dino_config:=/path/to/GroundingDINO_SwinT_OGC.py \
  grounding_dino_checkpoint:=/path/to/groundingdino_swint_ogc.pth \
  mobile_sam_checkpoint:=/path/to/mobile_sam.pt
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

---

## Package Layout

```
src/
├── openarm_sam_perception/           # Python node — AI perception
│   └── openarm_sam_perception/
│       └── sam_perception_node.py    # GroundingDINO + MobileSAM + depth → 3D point
│
└── openarm_perception_control/       # C++ node — arm orchestration
    ├── src/
    │   ├── control_node.cpp          # ControlNode: scan → segment → pick sequence
    │   └── main.cpp                  # Entry point + MultiThreadedExecutor
    ├── config/
    │   ├── scan_pose.yaml            # Joint angles for the overhead scan position
    │   ├── kinematics.yaml           # KDL IK solver config
    │   ├── joint_limits.yaml         # Per-joint velocity/acceleration limits
    │   ├── moveit_controllers.yaml   # Maps trajectory controllers to planning groups
    │   └── ros2_controllers.yaml     # ros2_control controller definitions
    ├── launch/
    │   └── servo_pipeline.launch.py  # Launches the full pipeline
    └── test/
        └── test_pick_object.py       # Manual integration test script
```

---

## Topic Reference

| Topic | Publisher | Subscriber | Type | Description |
|-------|-----------|------------|------|-------------|
| `/pick_prompt` | **you** | control_node | `std_msgs/String` | Your text description (e.g. `"red cup"`) — starts the pick sequence |
| `/segment_prompt` | control_node | sam_perception_node | `std_msgs/String` | Internal: relays your text to the AI node after the arm reaches scan pose |
| `/pick_target` | sam_perception_node | control_node | `geometry_msgs/PointStamped` | 3D object location in camera frame |

---

## Scan Pose Calibration

The scan pose (joint angles that position the camera directly above the workspace) is in `config/scan_pose.yaml`:

```yaml
scan_joint_values: [0.0, -0.3, 0.0, 1.6, 0.0, 1.57, 0.0]
#                   j1    j2   j3   j4   j5    j6    j7
```

Use `joint_state_publisher_gui` or the RViz joints panel to find the right values for your setup, then update this file before running pick tasks.

### Tuning Parameters

| Parameter | Default | File | Description |
|-----------|---------|------|-------------|
| `scan_joint_values` | `[0,−0.3,0,1.6,0,1.57,0]` | `scan_pose.yaml` | Overhead camera joint angles (7 values, radians) |
| `approach_height` | `0.15` m | `scan_pose.yaml` | Height above object before descending to pick |
| `planning_timeout` | `10.0` s | `scan_pose.yaml` | MoveIt OMPL time budget per plan call |
| `box_threshold` | `0.35` | launch file | GroundingDINO box confidence cutoff |
| `text_threshold` | `0.25` | launch file | GroundingDINO text-match confidence cutoff |
| `depth_scale` | `0.001` | launch file | RealSense raw depth unit → metres (mm → m) |

---

## Hardware

| Component | Details |
|-----------|---------|
| Robot | OpenArm v2.0 (7-DOF, right arm only) |
| Camera | Intel RealSense D-series, wrist-mounted on the right arm |
| GPU | Optional — GroundingDINO and MobileSAM fall back to CPU if no CUDA device is found |

---

## Known Issues

- **transformers >= 5.x**: `sam_perception_node` automatically patches `PreTrainedModel.get_head_mask` at startup if it is missing. No manual downgrade needed.
- **Controller timing**: The staggered startup delays (5 s / 7 s / 9 s) give `controller_manager` time to come up before controller spawners run. Increase these on slow machines by editing `servo_pipeline.launch.py`.
- **Slow startup**: Wait for `move_group ready` and `sam_perception_node: models loaded` in the terminal output before sending a `/pick_prompt`. Sending too early is silently ignored.

---

## Architecture Notes

### Node graph

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
           │  GroundingDINO             │  text prompt → bounding boxes
           │    └─► MobileSAM           │  bounding box → pixel mask
           │         └─► depth          │  mask + depth + K → 3D centroid
           │                            │
           │  Subscribes: /segment_prompt
           │  Publishes:  /pick_target
           └────────────┬───────────────┘
                        │  geometry_msgs/PointStamped
                        │  (3D point in camera optical frame)
                        │
           ┌────────────▼───────────────┐
           │   control_node             │  (C++)
           │                            │
           │  Subscribes: /pick_prompt  │  ← you send text here
           │  Publishes:  /segment_prompt│
           │  Subscribes: /pick_target  │
           │                            │
           │  Uses MoveIt move_group ──►│──► [MoveIt / right_arm]
           └────────────────────────────┘
```

### Threading model

`control_node` uses a `MultiThreadedExecutor` with two callback groups so that `on_pick_target()` can fire while `on_pick_prompt()` is blocked waiting for the AI result. A `SingleThreadedExecutor` would deadlock here.

```
Thread A (cbg_prompt_)                Thread B (cbg_target_)
──────────────────────────────        ──────────────────────────
on_pick_prompt() enters
  move_to_scan_pose() …waits…
  publish /segment_prompt
  wait_for_target():
    target_cv_.wait_for(30s) ─┐                sam_perception_node publishes
                               │   on_pick_target() fires
                               │     pending_target_ = *msg
                               │     target_cv_.notify_one()
    woken ◄────────────────────┘
  TF2 transform
  move_to_pick_pose()
```
