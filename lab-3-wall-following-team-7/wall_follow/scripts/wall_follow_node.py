#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import math

class WallFollow(Node):
    """ 
    Implement Wall Following on the car
    """
    def __init__(self):
        super().__init__('wall_follow_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # TODO: create subscribers and publishers
        self.scan_sub = self.create_subscription(LaserScan, lidarscan_topic, self.scan_callback, 10)
        self.scan_sub
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        # TODO: set PID gains
        self.kp = 4.5
        self.kd = 0.3
        self.ki = 0.05
        # self.kp = 1.5
        # self.kd = 0.05
        # self.ki = 0.15


        # TODO: store history
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

        self.time_gap = None

        # TODO: store any necessary values you think you'll need
        self.alpha = 0
        self.laser = None
        self.L = 1 # lookahead distance
        self.L_wall = 1 # our desired distance is 1 meter from the wall

    def get_range(self, range_data, angle):
        """
        Simple helper to return the corresponding range measurement at a given angle. Make sure you take care of NaNs and infs.

        Args:
            range_data: single range array from the LiDAR
            angle: between angle_min and angle_max of the LiDAR

        Returns:
            range: range measurement in meters at the given angle

        """
        #TODO: implement

        idx = int((angle - self.laser.angle_min) / self.laser.angle_increment)
        range_measurement = range_data[idx]
        if math.isnan(range_measurement) or math.isinf(range_measurement):
            range_measurement = self.laser.range_max    

        return range_measurement
    
    def get_error(self, range_data, dist):
        """
        Calculates the error to the wall. Follow the wall to the left (going counter clockwise in the Levine loop). You potentially will need to use get_range()

        Args:
            range_data: single range array from the LiDAR
            dist: desired distance to the wall

        Returns:
            error: calculated error
        """

        theta = np.deg2rad(45) # 0<theta<70deg
        a = self.get_range(range_data, np.pi/2 - theta)
        b = self.get_range(range_data, np.pi/2)

        self.alpha = np.arctan2(a * np.cos(theta) - b, a * np.sin(theta))
        Dt = b * np.cos(self.alpha)

        D_future = Dt + self.L * np.sin(self.alpha)
        error = dist - D_future

        return error


    def pid_control(self, error, velocity):
        """
        Based on the calculated error, publish vehicle control

        Args:
            error: calculated error
            velocity: desired velocity

        Returns:
            None
        """

        angle = 0.0

        if self.time_gap is None:
            self.prev_error = error
        
        else:
            # Use kp, ki & kd to implement a PID controller
            self.integral += error * self.time_gap

            P = self.kp * error
            D =  self.kd * (error - self.prev_error) / self.time_gap
            I = self.ki * self.integral

            angle = P + D + I


        drive_msg = AckermannDriveStamped()

        # Fill in drive message and publish

        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = -angle
        self.drive_pub.publish(drive_msg)
        self.prev_error = error

    def scan_callback(self, msg):
        """
        Callback function for LaserScan messages. Calculate the error and publish the drive message in this function.

        Args:
            msg: Incoming LaserScan message

        Returns:
            None
        """
        self.laser = msg
        time = msg.header.stamp.sec + (msg.header.stamp.nanosec / (10**9))
        if self.prev_time is not None:
            self.time_gap = time - self.prev_time

        error = self.get_error(np.array(msg.ranges), self.L_wall)
        
        if 0 < abs(self.alpha) <= np.pi/18:
            velocity = 1.5
        elif np.pi/18 < abs(self.alpha) <= np.pi/9:
            velocity = 1.0
        else:
            velocity = 0.5

        self.pid_control(error, velocity)

        self.prev_time = time


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    wall_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
