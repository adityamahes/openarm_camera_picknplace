import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def _find_models_dir():
    d = os.path.dirname(os.path.realpath(__file__))
    for _ in range(7):
        candidate = os.path.join(d, 'models')
        if os.path.isdir(candidate):
            return candidate
        d = os.path.dirname(d)
    return os.path.join(d, 'models')

def _default_dino_config():
    try:
        import groundingdino
        return os.path.join(
            os.path.dirname(groundingdino.__file__),
            'config', 'GroundingDINO_SwinT_OGC.py')
    except ImportError:
        return ''

_MODELS_DIR          = _find_models_dir()
_DEFAULT_DINO_CONFIG = _default_dino_config()
_DEFAULT_DINO_CKPT   = os.path.join(_MODELS_DIR, 'groundingdino_swint_ogc.pth')
_DEFAULT_SAM_CKPT    = os.path.join(_MODELS_DIR, 'mobile_sam.pt')


def _load_srdf(pkg_share):
    with open(os.path.join(pkg_share, 'config', 'openarm_right_arm.srdf'), 'r') as f:
        return f.read()


def _robot_and_moveit_spawner(context: LaunchContext, arm_type, robot_preset):
    arm_type_str  = context.perform_substitution(arm_type)
    preset_str    = context.perform_substitution(robot_preset)

    desc_share = get_package_share_directory('openarm_description')
    xacro_path = os.path.join(
        desc_share, 'assets', 'robot', 'openarm_v2.0', 'urdf', 'openarm_v20.urdf.xacro')

    xacro_mappings = {
        'arm_type':   arm_type_str,
        'robot_preset': preset_str,
        'collapse_internal_empty_links': 'true',
        'emit_grasp_frame': 'false',
    }

    robot_description = xacro.process_file(
        xacro_path, mappings=xacro_mappings,
    ).toprettyxml(indent='  ')

    pkg_share = get_package_share_directory('openarm_perception_control')

    moveit_config = (
        MoveItConfigsBuilder('openarm', package_name='openarm_bimanual_moveit_config')
        .robot_description(file_path=xacro_path, mappings=xacro_mappings)
        .robot_description_semantic(
            file_path=os.path.join(pkg_share, 'config', 'openarm_right_arm.srdf'))
        .robot_description_kinematics(
            file_path=os.path.join(pkg_share, 'config', 'kinematics.yaml'))
        .joint_limits(
            file_path=os.path.join(pkg_share, 'config', 'joint_limits.yaml'))
        .trajectory_execution(
            file_path=os.path.join(pkg_share, 'config', 'moveit_controllers.yaml'))
        .planning_pipelines(pipelines=['ompl'], default_planning_pipeline='ompl')
        .to_moveit_configs()
    )
    moveit_params = moveit_config.to_dict()

    rviz_cfg = os.path.join(
        get_package_share_directory('openarm_bimanual_moveit_config'),
        'config', 'openarm_v2.0', 'moveit.rviz',
    )

    return [
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            output='both',
            parameters=[
                {'robot_description': robot_description},
                os.path.join(pkg_share, 'config', 'ros2_controllers.yaml'),
            ],
        ),
        Node(
            package='moveit_ros_move_group',
            executable='move_group',
            output='screen',
            parameters=[moveit_params],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='log',
            arguments=['-d', rviz_cfg],
            parameters=[moveit_params],
        ),
    ]


def generate_launch_description():
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

    arm_type    = LaunchConfiguration('arm_type')
    robot_preset = LaunchConfiguration('robot_preset')

    robot_and_moveit = OpaqueFunction(
        function=_robot_and_moveit_spawner,
        args=[arm_type, robot_preset],
    )

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

    pkg_share = get_package_share_directory('openarm_perception_control')
    servo_node = Node(
        package='openarm_perception_control',
        executable='multi_rate_servo_node',
        name='multi_rate_servo_node',
        output='screen',
        parameters=[
            os.path.join(pkg_share, 'config', 'scan_pose.yaml'),
            {
                'robot_description_semantic': _load_srdf(pkg_share),
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

    return LaunchDescription([
        dino_config_arg,
        dino_ckpt_arg,
        sam_ckpt_arg,
        arm_type_arg,
        robot_preset_arg,
        robot_and_moveit,
        joint_state_broadcaster,
        arm_controller,
        gripper_controller,
        sam_node,
        servo_node,
    ])
