from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rokey_pjt'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
    # 필수 패키지 인덱스와 package.xml
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),

    # launch 파일 포함
    (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),

    # config, maps 파일 포함
    (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    (os.path.join('share', package_name, 'maps'), glob('maps/*.yaml') + glob('maps/*.pgm')),

    # rviz 파일 포함
    (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='weed',
    maintainer_email='jjoonmo0212@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'map_switcher = rokey_pjt.map_switcher:main',
            'bfs_path_finder0 = rokey_pjt.bfs_path_finder0:main',
            'bfs_path_finder1 = rokey_pjt.bfs_path_finder1:main',
            'pose_publisher0 = rokey_pjt.pose_publisher0:main',
            'pose_publisher1 = rokey_pjt.pose_publisher1:main',
            'dock_controller0 = rokey_pjt.dock_controller0:main',
            'dock_controller1 = rokey_pjt.dock_controller1:main',
            'waypoints_follower0 = rokey_pjt.waypoints_follower0:main',
            'waypoints_follower1 = rokey_pjt.waypoints_follower1:main',
            'nav0 = rokey_pjt.turtlebot4_0_nav:main',
            'path_marker0 = rokey_pjt.path_marker0:main',
            'path_marker1 = rokey_pjt.path_marker1:main',
            'controller0 = rokey_pjt.controller0:main',
            'controller1 = rokey_pjt.controller1:main',
        ],
    },
)
