from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ---------------------------------------------------------------------------
    # Arguments — override model paths on the command line, e.g.:
    #   ros2 launch openarm_perception_control servo_pipeline.launch.py \
    #     grounding_dino_config:=/path/to/cfg.py \
    #     grounding_dino_checkpoint:=/path/to/gdino.pth \
    #     mobile_sam_checkpoint:=/path/to/mobile_sam.pt
    # ---------------------------------------------------------------------------
    dino_config_arg = DeclareLaunchArgument(
        'grounding_dino_config',
        default_value='',
        description='Path to GroundingDINO config .py file',
    )
    dino_ckpt_arg = DeclareLaunchArgument(
        'grounding_dino_checkpoint',
        default_value='',
        description='Path to GroundingDINO checkpoint .pth file',
    )
    sam_ckpt_arg = DeclareLaunchArgument(
        'mobile_sam_checkpoint',
        default_value='',
        description='Path to MobileSAM checkpoint .pt file',
    )

    # ---------------------------------------------------------------------------
    # openarm_sam_perception  (Python – GroundingDINO + MobileSAM + depth)
    # ---------------------------------------------------------------------------
    sam_node = Node(
        package='openarm_sam_perception',
        executable='sam_perception_node',
        name='sam_perception_node',
        output='screen',
        parameters=[{
            'grounding_dino_config': LaunchConfiguration('grounding_dino_config'),
            'grounding_dino_checkpoint': LaunchConfiguration('grounding_dino_checkpoint'),
            'mobile_sam_checkpoint': LaunchConfiguration('mobile_sam_checkpoint'),
            'box_threshold': 0.35,
            'text_threshold': 0.25,
            # RealSense default topics (realsense2_camera driver)
            'rgb_topic': '/camera/color/image_raw',
            'depth_topic': '/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/color/camera_info',
            'depth_scale': 0.001,   # RealSense depth in mm → metres
        }],
    )

    # ---------------------------------------------------------------------------
    # openarm_perception_control  (C++ – MoveIt orchestration)
    # ---------------------------------------------------------------------------
    servo_node = Node(
        package='openarm_perception_control',
        executable='multi_rate_servo_node',
        name='multi_rate_servo_node',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('openarm_perception_control'),
                'config',
                'scan_pose.yaml',
            ])
        ],
    )

    return LaunchDescription([
        dino_config_arg,
        dino_ckpt_arg,
        sam_ckpt_arg,
        sam_node,
        servo_node,
    ])
