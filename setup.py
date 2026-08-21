import os
from glob import glob
from setuptools import setup

package_name = 'planning_ga_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.xml')),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'geometry_msgs',
        'nav_msgs',
        'numpy',                           # for CTE calculation
        'pandas',                          # for GA CSV loading
        'matplotlib',                      # for GA plotting (optional)
        'autoware_localization_msgs',      # for KinematicState
        'autoware_planning_msgs',          # for Trajectory
    ],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Genetic Algorithm Planner Node + Performance Tracker for Autoware.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 1. GA Global Planner (existing)
            'ga_planner_node = planning_ga_node.ga_planner_node:main',
            
            # 2. DWA Local Planner (existing)
            'dwa_local_planner_node = planning_ga_node.dwa_local_planner_node:main',
            
            # 3. NEW: Performance Tracker
            'performance_tracker = planning_ga_node.performance_tracker:main',
        ],
    },
)


