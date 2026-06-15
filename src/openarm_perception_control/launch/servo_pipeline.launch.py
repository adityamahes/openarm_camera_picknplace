# =============================================================================
# servo_pipeline.launch.py
# =============================================================================
# Launches the complete OpenArm pick-and-place pipeline on a single machine.
#
# Nodes started (in effective order):
#   1. robot_state_publisher      — publishes /robot_description and /tf from URDF
#   2. ros2_control_node          — hardware abstraction + controller manager
#   3. move_group                 — MoveIt 2 motion planner and execution server
#   4. rviz2                      — visualisation (uses the same MoveIt params)
#   5. joint_state_broadcaster    — ros2_control controller: publishes /joint_states
#      (delayed 5 s so controller_manager is ready)
#   6. right_joint_trajectory_controller — ros2_control arm controller
#      (delayed 7 s so joint_state_broadcaster is up first)
#   7. right_gripper_controller   — ros2_control gripper controller
#      (delayed 9 s so arm controller is confirmed active)
#   8. sam_perception_node        — Python: GroundingDINO + MobileSAM + depth → 3D point
#   9. control_node               — C++: orchestrates scan → segment → pick motion
#
# Launch arguments (all have defaults; override on the command line):
#   grounding_dino_config       path to GroundingDINO_SwinT_OGC.py
#   grounding_dino_checkpoint   path to groundingdino_swint_ogc.pth
#   mobile_sam_checkpoint       path to mobile_sam.pt
#   arm_type                    URDF xacro arm version (default: v20)
#   robot_preset                URDF xacro preset (default: right_arm_with_pinch_gripper)
# =============================================================================

import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


# -----------------------------------------------------------------------------
# _find_models_dir
# -----------------------------------------------------------------------------
# Walks up the directory tree from this launch file looking for a "models/"
# folder (up to 7 levels).  The models directory lives at the workspace root
# (~/openarm_ws/models/) and is found regardless of where the install tree
# puts this file.  Falls back to <workspace_root>/models if nothing is found.
def _find_models_dir():
    d = os.path.dirname(os.path.realpath(__file__))
    for _ in range(7):
        candidate = os.path.join(d, 'models')
        if os.path.isdir(candidate):
            return candidate
        d = os.path.dirname(d)
    return os.path.join(d, 'models')


# -----------------------------------------------------------------------------
# _default_dino_config
# -----------------------------------------------------------------------------
# Tries to locate the GroundingDINO config file inside the installed Python
# package (groundingdino/config/GroundingDINO_SwinT_OGC.py).
# Returns an empty string if the package is not installed — the user must then
# supply the path as a launch argument.
def _default_dino_config():
    try:
        import groundingdino
        return os.path.join(
            os.path.dirname(groundingdino.__file__),
            'config', 'GroundingDINO_SwinT_OGC.py')
    except ImportError:
        return ''


# Module-level constants computed once at import time so they are available
# to both the argument declarations and the node definitions below.
_MODELS_DIR          = _find_models_dir()
_DEFAULT_DINO_CONFIG = _default_dino_config()
_DEFAULT_DINO_CKPT   = os.path.join(_MODELS_DIR, 'groundingdino_swint_ogc.pth')
_DEFAULT_SAM_CKPT    = os.path.join(_MODELS_DIR, 'mobile_sam.pt')


# -----------------------------------------------------------------------------
# _load_srdf
# -----------------------------------------------------------------------------
# Reads the SRDF (Semantic Robot Description Format) file for the right arm
# as a raw string.  The SRDF defines planning groups, end-effectors, and
# collision exclusions — MoveIt needs it at runtime but cannot locate it from
# the package path alone when passed as a node parameter.
def _load_srdf(pkg_share):
    with open(os.path.join(pkg_share, 'config', 'openarm_right_arm.srdf'), 'r') as f:
        return f.read()


# -----------------------------------------------------------------------------
# _robot_and_moveit_spawner
# -----------------------------------------------------------------------------
# OpaqueFunction callback: called at launch time (not at parse time) so that
# arm_type and robot_preset — which are LaunchConfiguration substitutions —
# have been resolved to their final string values via context.perform_substitution.
#
# Returns a list of Node actions that are injected into the launch graph.
def _robot_and_moveit_spawner(context: LaunchContext, arm_type, robot_preset):
    # Resolve LaunchConfiguration substitutions to plain strings.
    arm_type_str = context.perform_substitution(arm_type)
    preset_str   = context.perform_substitution(robot_preset)

    desc_share = get_package_share_directory('openarm_description')
    # Path to the top-level URDF xacro file for the OpenArm v2.0.
    xacro_path = os.path.join(
        desc_share, 'assets', 'robot', 'openarm_v2.0', 'urdf', 'openarm_v20.urdf.xacro')

    # Xacro mappings are substitution variables used inside the .xacro file.
    # They control which variant of the URDF is generated.
    xacro_mappings = {
        'arm_type':   arm_type_str,
        'robot_preset': preset_str,
        # Merge empty intermediate links for cleaner TF trees.
        'collapse_internal_empty_links': 'true',
        # Omit the grasp frame link (not needed for this pipeline).
        'emit_grasp_frame': 'false',
    }

    # Process the xacro file to produce a plain URDF XML string.
    # toprettyxml formats it with 2-space indentation.
    robot_description = xacro.process_file(
        xacro_path, mappings=xacro_mappings,
    ).toprettyxml(indent='  ')

    pkg_share = get_package_share_directory('openarm_perception_control')

    # MoveItConfigsBuilder assembles all MoveIt configuration into a single dict.
    # Each method call loads a specific config file and merges it in.
    moveit_config = (
        MoveItConfigsBuilder('openarm', package_name='openarm_bimanual_moveit_config')
        # URDF — kinematic model of the robot.
        .robot_description(file_path=xacro_path, mappings=xacro_mappings)
        # SRDF — planning groups and collision matrix for the right arm.
        .robot_description_semantic(
            file_path=os.path.join(pkg_share, 'config', 'openarm_right_arm.srdf'))
        # kinematics.yaml — IK solver selection (KDL) and tolerances.
        .robot_description_kinematics(
            file_path=os.path.join(pkg_share, 'config', 'kinematics.yaml'))
        # joint_limits.yaml — per-joint velocity and acceleration limits.
        .joint_limits(
            file_path=os.path.join(pkg_share, 'config', 'joint_limits.yaml'))
        # moveit_controllers.yaml — maps trajectory controllers to planning groups.
        .trajectory_execution(
            file_path=os.path.join(pkg_share, 'config', 'moveit_controllers.yaml'))
        # Use OMPL as the only planner; set it as the default pipeline.
        .planning_pipelines(pipelines=['ompl'], default_planning_pipeline='ompl')
        .to_moveit_configs()
    )
    # to_dict() flattens all config sections into a single parameter dict
    # suitable for passing to ROS 2 node 'parameters' fields.
    moveit_params = moveit_config.to_dict()

    rviz_cfg = os.path.join(
        get_package_share_directory('openarm_bimanual_moveit_config'),
        'config', 'openarm_v2.0', 'moveit.rviz',
    )

    return [
        # --- robot_state_publisher ---
        # Reads /robot_description (URDF) and publishes /tf and /tf_static
        # containing every fixed and revolute joint transform.
        # joint_state_broadcaster feeds it with live /joint_states.
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),

        # --- ros2_control_node ---
        # The controller manager: owns the hardware interface (real or mock)
        # and hosts all ros2_control controllers.
        # Needs robot_description to know joint names and hardware plugins.
        # ros2_controllers.yaml defines which controllers are available.
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            output='both',
            parameters=[
                {'robot_description': robot_description},
                os.path.join(pkg_share, 'config', 'ros2_controllers.yaml'),
            ],
        ),

        # --- move_group ---
        # MoveIt 2 central node: exposes planning and execution actions,
        # maintains the planning scene, and connects to trajectory controllers.
        # Receives the full moveit_params dict (URDF, SRDF, IK, limits, etc.).
        Node(
            package='moveit_ros_move_group',
            executable='move_group',
            output='screen',
            parameters=[moveit_params],
        ),

        # --- rviz2 ---
        # Visualiser: shows the robot model, TF tree, planning scene, and
        # lets a developer interactively plan goals.
        # Output goes to 'log' (not screen) to keep terminal output readable.
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='log',
            arguments=['-d', rviz_cfg],
            parameters=[moveit_params],
        ),
    ]


# =============================================================================
# generate_launch_description
# =============================================================================
# Main entry point called by the launch system.
# Declares launch arguments and assembles the full node graph.
def generate_launch_description():

    # --- Launch argument declarations ---
    # DeclareLaunchArgument makes the argument available on the command line and
    # as a LaunchConfiguration substitution throughout the rest of the description.

    dino_config_arg = DeclareLaunchArgument(
        'grounding_dino_config',
        default_value=_DEFAULT_DINO_CONFIG,
        description='Path to GroundingDINO config .py file',
    )
    dino_ckpt_arg = DeclareLaunchArgument(
        'grounding_dino_checkpoint',
        default_value=_DEFAULT_DINO_CKPT,
        description='Path to GroundingDINO checkpoint .pth file',
    )
    sam_ckpt_arg = DeclareLaunchArgument(
        'mobile_sam_checkpoint',
        default_value=_DEFAULT_SAM_CKPT,
        description='Path to MobileSAM checkpoint .pt file',
    )
    arm_type_arg = DeclareLaunchArgument(
        'arm_type',
        default_value='v20',
        description='Arm version: v20 (openarm v2.0)',
    )
    robot_preset_arg = DeclareLaunchArgument(
        'robot_preset',
        default_value='right_arm_with_pinch_gripper',
        description='Robot preset for v2.0 (e.g. right_arm, right_arm_with_pinch_gripper)',
    )

    # LaunchConfiguration returns a substitution object that resolves to the
    # argument's string value at launch time (not at parse time).
    arm_type     = LaunchConfiguration('arm_type')
    robot_preset = LaunchConfiguration('robot_preset')

    # OpaqueFunction defers _robot_and_moveit_spawner until launch time so that
    # LaunchConfiguration substitutions (arm_type, robot_preset) have been
    # resolved to their final string values before the xacro file is processed.
    robot_and_moveit = OpaqueFunction(
        function=_robot_and_moveit_spawner,
        args=[arm_type, robot_preset],
    )

    # --- Controller spawners (delayed) ---
    # The spawner executable registers a controller with controller_manager.
    # Delays are staggered to avoid race conditions during startup:
    #   5 s  — give controller_manager time to start up.
    #   7 s  — wait for joint_state_broadcaster before starting the arm controller.
    #   9 s  — wait for the arm controller before starting the gripper controller.
    # --controller-manager-timeout 30 allows the spawner to retry for 30 s
    # if controller_manager is not yet answering (handles slow machines).

    joint_state_broadcaster = TimerAction(
        period=5.0,
        actions=[Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster', '-c', '/controller_manager',
                       '--controller-manager-timeout', '30'],
        )],
    )
    arm_controller = TimerAction(
        period=7.0,
        actions=[Node(
            package='controller_manager',
            executable='spawner',
            arguments=['right_joint_trajectory_controller', '-c', '/controller_manager',
                       '--controller-manager-timeout', '30'],
        )],
    )
    gripper_controller = TimerAction(
        period=9.0,
        actions=[Node(
            package='controller_manager',
            executable='spawner',
            arguments=['right_gripper_controller', '-c', '/controller_manager',
                       '--controller-manager-timeout', '30'],
        )],
    )

    # --- sam_perception_node ---
    # Python node that:
    #   1. Receives the text prompt on /segment_prompt.
    #   2. Runs GroundingDINO to find a bounding box.
    #   3. Runs MobileSAM to get a pixel mask.
    #   4. Back-projects the masked depth pixels to 3D using camera intrinsics.
    #   5. Publishes the result as a PointStamped on /pick_target.
    #
    # box_threshold / text_threshold: confidence cutoffs for GroundingDINO detections.
    # rgb_topic / depth_topic / camera_info_topic: Intel RealSense topic names.
    # depth_scale: converts raw uint16 depth values (mm) to metres (0.001).
    sam_node = Node(
        package='openarm_sam_perception',
        executable='sam_perception_node',
        name='sam_perception_node',
        output='screen',
        parameters=[{
            'grounding_dino_config':      LaunchConfiguration('grounding_dino_config'),
            'grounding_dino_checkpoint':  LaunchConfiguration('grounding_dino_checkpoint'),
            'mobile_sam_checkpoint':      LaunchConfiguration('mobile_sam_checkpoint'),
            'box_threshold':  0.35,
            'text_threshold': 0.25,
            'rgb_topic':         '/camera/color/image_raw',
            'depth_topic':       '/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/color/camera_info',
            'depth_scale': 0.001,
        }],
    )

    # --- control_node ---
    # C++ node that orchestrates the pick sequence.
    # Loaded with two parameter sources (merged in order):
    #   1. scan_pose.yaml — overrides scan_joint_values, approach_height, planning_timeout.
    #   2. Inline dict   — provides robot_description_semantic (SRDF text) and
    #                      robot_description_kinematics (KDL IK solver config) so
    #                      the node can instantiate MoveGroupInterface without
    #                      relying on a separate SRDF node.
    pkg_share = get_package_share_directory('openarm_perception_control')
    control_node = Node(
        package='openarm_perception_control',
        executable='control_node',
        name='control_node',
        output='screen',
        parameters=[
            os.path.join(pkg_share, 'config', 'scan_pose.yaml'),
            {
                # SRDF must be passed as a string parameter because MoveGroupInterface
                # reads it from the /robot_description_semantic parameter, not a file.
                'robot_description_semantic': _load_srdf(pkg_share),
                # KDL IK solver settings for the right_arm planning group.
                'robot_description_kinematics': {
                    'right_arm': {
                        'kinematics_solver': 'kdl_kinematics_plugin/KDLKinematicsPlugin',
                        'kinematics_solver_search_resolution': 0.005,
                        'kinematics_solver_timeout': 0.005,
                    }
                },
            },
        ],
    )

    # Assemble the full launch description.
    # Order matters only for the argument declarations (they must precede their
    # first use); actual node startup order is controlled by the TimerActions.
    return LaunchDescription([
        dino_config_arg,
        dino_ckpt_arg,
        sam_ckpt_arg,
        arm_type_arg,
        robot_preset_arg,
        robot_and_moveit,           # robot_state_publisher, ros2_control_node, move_group, rviz2
        joint_state_broadcaster,    # +5 s
        arm_controller,             # +7 s
        gripper_controller,         # +9 s
        sam_node,
        control_node,
    ])
