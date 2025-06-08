from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rokey_pjt',
            executable='dock_controller0',
            name='dock_controller0',
            namespace='robot0',
            output='screen'
        ),
        Node(
            package='rokey_pjt',
            executable='waypoints_follower0',
            name='waypoints_follower0',
            namespace='robot0',
            output='screen'
        ),
        Node(
            package='rokey_pjt',
            executable='pose_publisher0',
            name='pose_publisher0',
            namespace='robot0',
            output='screen'
        ),
        Node(
            package='rokey_pjt',
            executable='bfs_path_finder0',
            name='bfs_path_finder0',
            namespace='robot0',
            output='screen'
        ),
        Node(
            package='rokey_pjt',
            executable='controller0',
            name='controller0',
            namespace='robot0',
            output='screen'
        ),
    ])

