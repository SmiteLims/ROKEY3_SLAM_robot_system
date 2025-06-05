from setuptools import find_packages, setup

package_name = 'object_detection_module'

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
    maintainer='smite',
    maintainer_email='smitelims15@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_detection_cam0 = object_detection_module.yolo_detection_cam0:main',
            'yolo_ocr_depth_checker0 = object_detection_module.yolo_ocr_depth_checker0:main',
            'yolo_detection_cam1 = object_detection_module.yolo_detection_cam1:main',
            'yolo_ocr_depth_checker1 = object_detection_module.yolo_ocr_depth_checker1:main',

        ],
    },
)
