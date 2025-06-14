from setuptools import find_packages, setup

package_name = 'system_controller'

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
            "system_controller = system_controller.system_controller:main",
            "system_controller_DB = system_controller.system_controller_DB:main",
            "system_controller_noDB = system_controller.system_controller_noDB:main",


        ],
    },
)
