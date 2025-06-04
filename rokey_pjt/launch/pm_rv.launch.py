from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        # path_marker0 노드 실행
        Node(
            package='rokey_pjt',
            executable='path_marker0',
            name='path_marker0',
            namespace='robot0',
            output='screen'
        ),

        Node(
            package='rokey_pjt',
            executable='path_marker1',
            name='path_marker1',
            namespace='robot1',
            output='screen'
        ),

        # RViz 실행
        ExecuteProcess(
            cmd=['rviz2', '-d', '/home/rokey/test_ws/src/rokey_pjt/rviz/rviz.rviz'],
            output='screen'
        )
    ])
