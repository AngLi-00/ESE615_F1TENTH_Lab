#!/usr/bin/env python3
# NOTE: You have to add the shebang line above as the first thing in this file!

# Criteria:
# @ relay subscribes to the drive topic.
# @ In the subscriber callback, take the speed and steering 
# angle from the incoming message, multiply both by 3, and 
# publish the new values via another AckermannDriveStamped 
# message to a topic named drive_relay.



import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ackermann_msgs.msg import AckermannDriveStamped



class RelayNode(Node):
    def __init__(self):
        super().__init__('relay')
        self.subscription = self.create_subscription(
            AckermannDriveStamped,
            'drive',
            self.listener_callback,
            10)
        self.subscription  # 防止未使用的变量警告

        self.publisher_ = self.create_publisher(
            AckermannDriveStamped,
            'drive_relay',
            10)

    def listener_callback(self, msg):
        new_msg = AckermannDriveStamped()
        new_msg.drive.speed = msg.drive.speed * 3
        new_msg.drive.steering_angle = msg.drive.steering_angle * 3

        self.publisher_.publish(new_msg)
        self.get_logger().info('Speed: "%s", Steering Angle: "%s"' % (new_msg.drive.speed, new_msg.drive.steering_angle))

def main(args=None):
    rclpy.init(args=args)
    relay_node = RelayNode()
    rclpy.spin(relay_node)
    relay_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()