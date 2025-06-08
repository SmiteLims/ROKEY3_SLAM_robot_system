from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rokey_pjt',
            executable='dock_controller1',
            name='dock_controller1',
            namespace='robot1',
            output='screen'
        ),
        Node(
            package='rokey_pjt',
            executable='waypoints_follower1',
            name='waypoints_follower1',
            namespace='robot1',
            output='screen'
        ),
        Node(
            package='rokey_pjt',
            executable='pose_publisher1',
            name='pose_publisher1',
            namespace='robot1',
            output='screen'
        ),
        Node(
            package='rokey_pjt',
            executable='bfs_path_finder1',
            name='bfs_path_finder1',
            namespace='robot1',
            output='screen'
        ),
        Node(
            package='rokey_pjt',
            executable='controller1',
            name='controller1',
            namespace='robot1',
            output='screen'
        ),
    ])

