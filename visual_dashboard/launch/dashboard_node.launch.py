from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='visual_dashboard',       
            executable='dashboard_node', 
            name='multi_cam_dashboard',  
            output='screen',
            remappings=[('/robot0/action', '/robot0/action_dashboard'),
                        ('/robot1/action', '/robot1/action_dashboard'),
                        ]
            # parameters=[{'param_name': param_value}],  # 파라미터가 필요하면 추가
        ),
    ])
