from setuptools import setup

package_name = 'openarm_sam_perception'

pip_requirements = ['groundingdino-py', 'mobile-sam']

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'] + pip_requirements,
    zip_safe=True,
    maintainer='aditya',
    maintainer_email='4aditya.m@gmail.com',
    description='Grounded SAM perception node for OpenArm pick-and-place',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sam_perception_node = openarm_sam_perception.sam_perception_node:main',
        ],
    },
)
