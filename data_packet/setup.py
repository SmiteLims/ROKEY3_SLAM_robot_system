from setuptools import find_packages, setup
import glob
import os

package_name = 'data_packet'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ethica',
    maintainer_email='ethica.the@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'action_mux = data_packet.action_mux:main',
            'rviz_graphic_robot0 = data_packet.rviz_graphic_robot0:main',
            'rviz_graphic_robot1 = data_packet.rviz_graphic_robot1:main',
        ],
    },
)
