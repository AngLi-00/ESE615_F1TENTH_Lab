from launch import LaunchDescription
from launch_ros.actions import Node

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    v_param = DeclareLaunchArgument(
        'v', default_value='1.0',
        description='Velocity parameter for talker node')

    d_param = DeclareLaunchArgument(
        'd', default_value='0.5',
        description='Steering angle parameter for talker node')

    talker_node = Node(
        package='lab1_pkg',
        executable='talker',
        parameters=[{'v': LaunchConfiguration('v'), 
                     'd': LaunchConfiguration('d')}]
    )

    relay_node = Node(
        package='lab1_pkg',
        executable='relay.py'
    )

    # Create the launch description and populate
    ld = LaunchDescription()
    ld.add_action(v_param)
    ld.add_action(d_param)
    ld.add_action(talker_node)
    ld.add_action(relay_node)

    return ld
