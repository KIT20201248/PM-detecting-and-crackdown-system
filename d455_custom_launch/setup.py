from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'd455_custom_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'params'), glob('params/*.yaml')),  
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dh',
    maintainer_email='dh@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'd455_record = d455_custom_launch.d455_record:main'
        ],
    },
)
