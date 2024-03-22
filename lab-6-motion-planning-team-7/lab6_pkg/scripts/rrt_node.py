#!/usr/bin/env python3
"""
This file contains the class definition for tree nodes and RRT
Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
"""
import numpy as np
from numpy import linalg as LA
import math
import copy
import csv
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import OccupancyGrid

# TODO: import as you need
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray
from os.path import expanduser
import time

# class def for tree nodes
# It's up to you if you want to use this
class Tree_Node(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = None # only used in RRT*
        self.is_root = False

# not currently being used
class OccGrid(object):
    def __init__(self, res, width, height):
        self.res = res 
        self.w = width
        self.h = height


# class def for RRT
class RRT(Node):
    def __init__(self):
        """
        start (numpy.ndarray or (x, y)): Start Position [x,y], the root of the tree and also the current position of the vehicle
        goal (numpy.ndarray or (x, y)): Goal Position [x,y]
        """        
        super().__init__('rrt_node')
        self.start_point = np.array([0, 0])
        self.goal_point = np.array([0, 0])
        self.goal_dist = 0.0
        self.start_node = Tree_Node(self.start_point[0], self.start_point[1])
        self.goal_node = Tree_Node(self.goal_point[0], self.goal_point[1])
        self.tree = []
        self.waypoints = None
        self.np_waypoints = None
        self.pose = None
        self.posX = None
        self.posY = None

        self.sim = True

        self.og_sample_point = None

        # topics, not saved as attributes
        # TODO: grab topics from param file, you'll need to change the yaml file

        scan_topic = "/scan"
        if not self.sim:
            pose_topic = 'pf/viz/inferred_pose'
        else:
            pose_topic = '/ego_racecar/odom'

        # you could add your own parameters to the rrt_params.yaml file,
        # and get them here as class attributes as shown above.
        # self.occ_grid = OccGrid(0.1, 3.0, 3.0)

        # TODO: create subscribers
        self.scan_sub_ = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            1)
        self.scan_sub_


        self.pose_sub_ = self.create_subscription(
            PoseStamped if not self.sim else Odometry,
            pose_topic,
            self.pose_callback,
            1)
        self.pose_sub_

        # publishers
        # TODO: create a drive message publisher, and other publishers that you might need
        self.viz_pub_ = self.create_publisher(MarkerArray, '/visualization_nodes', 10)
        # self.viz_target_pub_ = self.create_publisher(Marker, '/visualization_target', 10)
        self.viz_RRTgoal_pub_ = self.create_publisher(Marker, '/visualization_RRTgoal', 10)
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.viz_og_pub_ = self.create_publisher(OccupancyGrid, '/visualization_og', 10)
        # self.viz_sample_pub_ = self.create_publisher(Marker, '/visualization_sample_point', 10)
        self.viz_pub_tree_ = self.create_publisher(MarkerArray, '/visualization_tree', 10)
        # self.viz_one_node_pub_ = self.create_publisher(Marker, '/visualization_one_node', 10)
        # self.viz_nearest_node_pub_ = self.create_publisher(Marker, '/visualization_nearest_node', 10)
        # self.viz_one_point_pub_ = self.create_publisher(Marker, '/visualization_one_point', 10)

        # class attributes
        # TODO: maybe create your occupancy grid here

        # parameters
        self.expand_dis = 0.4

        self.occupancy_grid = np.ones((36, 100)) * -1.0
        self.lidar_max = 6.0
        self.is_goal_threshold = 0.2
        self.resolution = 0.1
        self.max_iter = 400
        self.Kp = 0.35
        self.steering_min = -math.pi*2/3
        self.steering_max = math.pi*2/3
        self.angle_min = 0.0
        self.angle_max = 0.0
        self.L = 2.0 # pure_pursuit

        home = expanduser('~')
        # folder_path = '/home/nvidia/f1tenth_ws/src/lab6_pkg/csv_files/'
        folder_path = '/home/angli/sim_ws/src/lab6_pkg/csv_files/'
        # folder_path = '/home/sidpan/Repos/f1tenth_lab6/'
        filename = 'wp-3-21-loop.csv'
        self.csv_file_path = os.path.join(folder_path, filename)

        goalpoints = []
        with open(self.csv_file_path, mode='r') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader, None) # skip the headers
            for row in csv_reader:
                goalpoints.append({
                    'posX': float(row[0]),
                    'posY': float(row[1])
                })
        self.goalpoints = goalpoints
        self.np_goalpoints = np.array([np.array([wp['posX'], wp['posY']]) for wp in self.goalpoints])
        self.L_goal = 3.0 # lookahead distance for goalpoints

    def scan_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here

        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:

        """
        # print("Scan Callback")

        self.angle_increment = scan_msg.angle_increment
        self.angle_max = scan_msg.angle_max
        self.angle_min = scan_msg.angle_min
        # print("Angle Min: ", self.angle_min, "Angle Max: ", self.angle_max)
        ranges, angles = self.preprocess_lidar(scan_msg.ranges)

        x_detect = ranges * np.cos(angles)
        y_detect = ranges * np.sin(angles)

        x_coords = np.round(x_detect / self.resolution) + self.occupancy_grid.shape[1] // 2
        y_coords = np.round(y_detect / self.resolution) + self.occupancy_grid.shape[0] // 2

        x_coords = x_coords.astype(int)
        y_coords = y_coords.astype(int)

        # occupancy grid update
        self.occupancy_grid.fill(0)
                
        filter = (x_coords >= 0) & (x_coords < self.occupancy_grid.shape[1]) & \
                 (y_coords >= 0) & (y_coords < self.occupancy_grid.shape[0]) 

        self.occupancy_grid[y_coords[filter], x_coords[filter]] = 1
        # if self.og_sample_point is not None:
        #     self.occupancy_grid[self.og_sample_point[1], self.og_sample_point[0]] = -1

        self.visualize_occ_grid()


    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        ranges = np.array(ranges)
        angles = np.arange(self.angle_min, self.angle_max, self.angle_increment)

        # remove indices where lidar is inf or too far
        idx = (~np.isfinite(ranges)) | (ranges > self.lidar_max)
        proc_ranges = ranges[~ idx]
        proc_angles = angles[~ idx]
        
        return proc_ranges, proc_angles
    
    def visualize_occ_grid(self):
        if self.pose is None:
            return
        

        viz_og = OccupancyGrid()
        viz_og.header.frame_id = "map"

        int_grid = self.occupancy_grid.astype(np.int8)
        viz_og.data = np.ravel(int_grid, order='C').tolist()

        viz_og.info.origin = self.pose
        vec_length = math.sqrt((self.occupancy_grid.shape[1] // 2 * self.resolution) ** 2 + 
                                (self.occupancy_grid.shape[0] // 2 * self.resolution) ** 2)
        
        to_center_angle = np.arctan2((self.occupancy_grid.shape[0] // 2 * self.resolution), 
                                     (self.occupancy_grid.shape[1] // 2 * self.resolution))
        
        vec_angle = to_center_angle + self.theta
        viz_og.info.origin.position.x = self.pose.position.x - vec_length * math.cos(vec_angle)
        viz_og.info.origin.position.y = self.pose.position.y - vec_length * math.sin(vec_angle)

        viz_og.info.width = self.occupancy_grid.shape[1]
        viz_og.info.height = self.occupancy_grid.shape[0]
        viz_og.info.resolution = self.resolution

        self.viz_og_pub_.publish(viz_og)

        return

    def pose_callback(self, pose_msg):
        """
        The pose callback when subscribed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        Returns:

        """
        # print("Pose Callback")

        if self.sim:
            self.posX = pose_msg.pose.pose.position.x
            self.posY = pose_msg.pose.pose.position.y
            quat = pose_msg.pose.pose.orientation
            self.pose = pose_msg.pose.pose
        else:
            self.posX = pose_msg.pose.position.x
            self.posY = pose_msg.pose.position.y
            quat = pose_msg.pose.orientation
            self.pose = pose_msg.pose

        quat = [quat.x, quat.y, quat.z, quat.w]
        euler = euler_from_quaternion(quat)
        theta = euler[2]
        self.theta = theta
        
        # Calculate the RRTgoalpoint to track
        rrt_goal_pos = (self.posX + self.L_goal*math.cos(theta), self.posY + self.L_goal*math.sin(theta))
        np_current_pos = np.zeros_like(self.np_goalpoints)
        np_current_pos[:, 0] = rrt_goal_pos[0]
        np_current_pos[:, 1] = rrt_goal_pos[1]
        dist_rrt = np.linalg.norm(self.np_goalpoints - np_current_pos, axis=1)
        closest_gp_index = np.argmin(dist_rrt)
        closest_gp = self.goalpoints[closest_gp_index]

        self.visualize_RRTgoalpoint(self.goalpoints, closest_gp_index)

        # Set the RRT goal point
        self.goal_point = np.array([closest_gp['posX'], closest_gp['posY']])

        # Set start point
        self.start_point = np.array([self.posX, self.posY])
        self.start_node = Tree_Node(self.start_point[0], self.start_point[1])
        self.goal_dist = np.linalg.norm(self.start_point - self.goal_point)

        # print("Start: ", self.start_point)
        # print("Goal: ", self.goal_point) 




        # ##################################################
        # ############## test collision check ##############
        # ##################################################
        # self.waypoints = None
        # # self.start_node = Tree_Node(0.0, 0.0)
        # # self.goal_node = Tree_Node(5.0, -1.0)
        # self.start_node = Tree_Node(0.0, 0.0)
        # # self.goal_node = Tree_Node(3.2, -0.8)
        # self.goal_node = Tree_Node(2.0, 0.7)
        
        # self.visualize_nearest_node(self.start_node)
        # self.visualize_one_node(self.goal_node)

        # ####coordinate transformation test####
        # print("world2og",self.world_to_og_coords(self.goal_node.x, self.goal_node.y))
        # aaa, bbb = self.world_to_og_coords(self.goal_node.x, self.goal_node.y)
        # self.og_sample_point = (aaa, bbb)
        # print("og2world",self.og_coords_to_world(aaa, bbb))

        # print("check collision result: ",self.check_collision(self.start_node, self.goal_node))
        # ##################################################
        # ##################################################
        # ##################################################







        # Run RRT
        self.waypoints = self.rrt_planning()

        if self.waypoints is None:
            print("No path found")
            ######
            # Publish drive message
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 0.1
            drive_msg.drive.steering_angle = 0.0

            self.drive_pub_.publish(drive_msg)
        else:
            print("Path found")
            self.np_waypoints = np.array([np.array([wp.x, wp.y]) for wp in self.waypoints])

            self.visualize_nodes(self.np_waypoints, self.tree)

            # Calculate the current waypoint to track
            lookahead_pos = (self.posX + self.L*math.cos(theta), self.posY + self.L*math.sin(theta))
            np_current_pos = np.zeros_like(self.np_waypoints)
            np_current_pos[:, 0] = lookahead_pos[0]
            np_current_pos[:, 1] = lookahead_pos[1]
            dist = np.linalg.norm(self.np_waypoints - np_current_pos, axis=1)
            closest_wp_index = np.argmin(dist)
            closest_wp = self.waypoints[closest_wp_index]

            # self.visualize_waypoint(self.waypoints, closest_wp_index)

            # Transform goal point to vehicle frame of reference
            x_diff = closest_wp.x - self.posX
            y_diff = closest_wp.y - self.posY       
            x_goal_car = x_diff * math.cos(theta) + y_diff * math.sin(theta)
            y_goal_car = -x_diff * math.sin(theta) + y_diff * math.cos(theta)

            # Run pure pursuit
            steering_angle = self.pure_pursuit((x_goal_car, y_goal_car))

            # Publish drive message
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 0.3
            drive_msg.drive.steering_angle = steering_angle
            self.drive_pub_.publish(drive_msg)
        return None

    def rrt_planning(self):
        """
        The main rrt planning method
        Args:
        Returns:
            path ([]): a list of Tree_Nodes connecting the start and goal
        """

        self.tree = [self.start_node]

        for i in range(self.max_iter):
            # print("RRT Planning...", i)
            # Sample a point in the free space
            sampled_point = self.sample()
            # print("sample done")

            # Find the nearest node to the sampled point
            nearest_node_index = self.nearest(self.tree, sampled_point)
            nearest_node = self.tree[nearest_node_index]
            # print("nearest_node:", nearest_node.x, nearest_node.y)
            # self.visualize_nearest_node(nearest_node)
            # Steer the nearest node to the sampled point
            new_node = self.steer(nearest_node, sampled_point)
            # print("new_node: ", new_node.x, new_node.y)
            # self.visualize_one_node(new_node)

            # Check for collision
            if self.check_collision(nearest_node, new_node):
                # print("Collision Free")
                # Add the new node to the tree
                self.tree.append(new_node)
                # tree_node_size = len(self.tree)
                # print("Tree Size: ", tree_node_size)
                # print("collision free new node: ",new_node.x, new_node.y)
                # self.visualize_one_node(new_node)
                # for i in range(len(self.tree)):
                #     print("@ Tree Node", i, " : ", self.tree[i].x, self.tree[i].y)
                

                # Check if the new node is close enough to the goal
                if self.is_goal(new_node, self.goal_point[0], self.goal_point[1]):
                    # print("Close enough to goal")
                    # If the new node is close enough to the goal, return the path
                    return self.find_path(self.tree, new_node)

        # tree_node_size = len(self.tree)
        # print("Tree Size: ", tree_node_size)
        # for i in range(len(self.tree)):
        #     print("Tree Node", i, " : ", self.tree[i].x, self.tree[i].y)
        
        # self.visualize_tree(self.tree)
        return None
    
    def visualize_Samplepoint(self, samplepoint):

        """samplepoint: (x, y) (float float): a tuple representing the sampled point"""
        sample_marker = Marker()
        sample_marker.header.frame_id = "map"
        sample_marker.id = 0
        sample_marker.type = Marker.SPHERE
        sample_marker.action = Marker.ADD
        sample_marker.scale.x = 0.15
        sample_marker.scale.y = 0.15
        sample_marker.scale.z = 0.5
        sample_marker.color.a = 1.0  # Alpha is set to 1 to ensure the marker is not transparent
        sample_marker.color.r = 0.0 
        sample_marker.color.g = 0.0
        sample_marker.color.b = 0.0 
        sample_marker.pose.orientation.w = 1.0
        sample_marker.pose.position.x = samplepoint[0]
        sample_marker.pose.position.y = samplepoint[1]
        sample_marker.pose.position.z = 0.0
        # marker.lifetime = rclpy.duration.Duration()  # Setting to zero so it never auto-deletes

        self.viz_sample_pub_.publish(sample_marker)
    
    def visualize_RRTgoalpoint(self, goalpoints, current_RRTgoalpoint_index):

        waypoint = goalpoints[current_RRTgoalpoint_index]
        target_marker = Marker()
        target_marker.header.frame_id = "map"
        target_marker.id = 0
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.scale.x = 0.15
        target_marker.scale.y = 0.15
        target_marker.scale.z = 0.5
        target_marker.color.a = 1.0  # Alpha is set to 1 to ensure the marker is not transparent
        target_marker.color.r = 255.0 
        target_marker.color.g = 0.0
        target_marker.color.b = 0.0 
        target_marker.pose.orientation.w = 1.0
        target_marker.pose.position.x = waypoint['posX']
        target_marker.pose.position.y = waypoint['posY']
        target_marker.pose.position.z = 0.0
        # marker.lifetime = rclpy.duration.Duration()  # Setting to zero so it never auto-deletes

        self.viz_RRTgoal_pub_.publish(target_marker)
    
    def visualize_nodes(self, points, tree_nodes):
        marker_array = MarkerArray()
        for i, point in enumerate(points):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.5
            marker.color.a = 1.0  # Alpha is set to 1 to ensure the marker is not transparent
            marker.color.r = 1.0  
            marker.color.g = 1.0
            marker.color.b = 1.0  
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = 0.0
            # marker.lifetime = rclpy.duration.Duration()  # Setting to zero so it never auto-deletes
            marker_array.markers.append(marker)

        # # visualize lines for each nodes with its parent node
        # line_id_offset = len(points)
        # for i, node in enumerate(tree_nodes):
        #     if node.parent is not None:
        #         line_marker = Marker()
        #         line_marker.header.frame_id = "map"
        #         line_marker.id = i + line_id_offset
        #         line_marker.type = Marker.LINE_STRIP
        #         line_marker.action = Marker.ADD
        #         line_marker.scale.x = 0.05
        #         line_marker.color.a = 1.0
        #         line_marker.color.r = 0.0
        #         line_marker.color.g = 0.0
        #         line_marker.color.b = 1.0
        #         line_marker.points.append(Point(x = node.parent.x, y = node.parent.y, z = 0.0))
        #         line_marker.points.append(Point(x = node.x, y = node.y, z = 0.0))
        #         marker_array.markers.append(line_marker)

        self.viz_pub_.publish(marker_array)

    def visualize_waypoint(self, waypoints, current_waypoint):

        waypoint = waypoints[current_waypoint]
        target_marker = Marker()
        target_marker.header.frame_id = "map"
        target_marker.id = 0
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.scale.x = 0.15
        target_marker.scale.y = 0.15
        target_marker.scale.z = 0.5
        target_marker.color.a = 1.0  # Alpha is set to 1 to ensure the marker is not transparent
        target_marker.color.r = 0.0
        target_marker.color.g = 255.0
        target_marker.color.b = 0.0 
        target_marker.pose.orientation.w = 1.0
        target_marker.pose.position.x = waypoint.x
        target_marker.pose.position.y = waypoint.y
        target_marker.pose.position.z = 0.0
        # marker.lifetime = rclpy.duration.Duration()  # Setting to zero so it never auto-deletes

        self.viz_target_pub_.publish(target_marker)

    def visualize_tree(self, tree_nodes):
        marker_array = MarkerArray()
        for i, tree_node in enumerate(tree_nodes):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.5
            marker.color.a = 1.0  # Alpha is set to 1 to ensure the marker is not transparent
            marker.color.r = 100.0  
            marker.color.g = 100.0
            marker.color.b = 0.0  
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = tree_node.x
            marker.pose.position.y = tree_node.y
            marker.pose.position.z = 0.0
            # marker.lifetime = rclpy.duration.Duration()  # Setting to zero so it never auto-deletes
            marker_array.markers.append(marker)

        self.viz_pub_tree_.publish(marker_array)

    def visualize_one_node(self, node):

        marker = Marker()
        marker.header.frame_id = "map"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.5
        marker.color.a = 1.0  # Alpha is set to 1 to ensure the marker is not transparent
        marker.color.r = 100.0 
        marker.color.g = 0.0
        marker.color.b = 100.0 
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = node.x
        marker.pose.position.y = node.y
        marker.pose.position.z = 0.0
        # marker.lifetime = rclpy.duration.Duration()  # Setting to zero so it never auto-deletes

        self.viz_one_node_pub_.publish(marker)

    def visualize_nearest_node(self, node):

        marker = Marker()
        marker.header.frame_id = "map"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.5
        marker.color.a = 1.0  # Alpha is set to 1 to ensure the marker is not transparent
        marker.color.r = 150.0 
        marker.color.g = 100.0
        marker.color.b = 250.0 
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = node.x
        marker.pose.position.y = node.y
        marker.pose.position.z = 0.0
        # marker.lifetime = rclpy.duration.Duration()  # Setting to zero so it never auto-deletes

        self.viz_nearest_node_pub_.publish(marker)

    def visualize_one_point(self, point):
        """
        visualize a point(x,y) in rviz
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.5
        marker.color.a = 1.0  # Alpha is set to 1 to ensure the marker is not transparent
        marker.color.r = 0.0 
        marker.color.g = 100.0
        marker.color.b = 100.0 
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = 0.0
        # marker.lifetime = rclpy.duration.Duration()  # Setting to zero so it never auto-deletes

        self.viz_one_point_pub_.publish(marker)

    def pure_pursuit(self, goal_pose):
        '''
        Returns a steering angle given a goal pose in terms of the car's reference frame.

        goal_pose: Tuple (x, y) of the goal pose
        return: Float of desired steering angle
        '''

        L2 = (goal_pose[0] ** 2) + (goal_pose[1] ** 2)
        gamma = 2 * abs(goal_pose[1]) / L2
        angle = self.Kp * gamma

        angle *= np.sign(goal_pose[1])

        angle = max(self.steering_min, angle)
        angle = min(self.steering_max, angle)

        return angle

    def sample(self):
        """
        This method should randomly sample the free space, and returns a viable point

        Args:
        Returns:
            (x, y) (float float): a tuple representing the sampled point

        """     
        while True:

            angle = np.random.uniform(self.steering_min, self.steering_max)
            dist = np.random.uniform(0.8, self.goal_dist)

            x = self.posX + dist * math.cos(angle + self.theta)
            y = self.posY + dist * math.sin(angle + self.theta)

            grid_x, grid_y = self.world_to_og_coords(x, y)

            grid_x, grid_y = int(grid_x), int(grid_y)
            # x_adj, y_adj = self.og_coords_to_world(grid_x, grid_y)
            # self.visualize_Samplepoint((x_adj, y_adj))

            #print("Angle:", angle)
            #print("x:", x, "y:", y)
            # print("Grid X:", grid_x, "Grid Y:", grid_y)
            #print("Occupancy Grid Value:", self.occupancy_grid[grid_y, grid_x])

            if (0 <= grid_x < self.occupancy_grid.shape[1]) and (0 <= grid_y < self.occupancy_grid.shape[0]) and (self.occupancy_grid[grid_y, grid_x] == 0):
                # print("sampled point: ", grid_x, grid_y)
                # print("Goal_Dist: ", self.goal_dist)
                # real_x, real_y = self.og_coords_to_world(grid_x, grid_y)
                # real_x = (grid_x - self.occupancy_grid.shape[1] / 2) * self.resolution
                # real_y = (grid_y - self.occupancy_grid.shape[0] / 2) * self.resolution
                # self.og_sample_point = (grid_x, grid_y)
                # self.visualize_Samplepoint((x, y))
                break

        # return (grid_x, grid_y)
        # print("feasible point: ", x, y)
        return (x, y)

    def nearest(self, tree, sampled_point):
        """
        This method should return the nearest node on the tree to the sampled point

        Args:
            tree ([]): the current RRT tree
            sampled_point (tuple of (float, float)): point sampled in free space
        Returns:
            nearest_node (int): index of neareset node on the tree
        """
        #nearest_node = 0.0
        #min_dist = float('inf')

        tree_xy = np.array([np.array([n.x, n.y]) for n in tree])
        np_sampled = np.zeros_like(tree_xy)
        np_sampled[:, 0] = sampled_point[0]
        np_sampled[:, 1] = sampled_point[1]
        dist = np.linalg.norm(tree_xy - np_sampled, axis=1)
        nearest_node = np.argmin(dist)


        # for i in range(len(tree)):
        #     x_diff = tree[i].x - sampled_point[0]
        #     y_diff = tree[i].y - sampled_point[1]
        #     dist = math.sqrt(x_diff**2 + y_diff**2) 

        #     if dist < min_dist:
        #         min_dist = dist
        #         nearest_node = i

        return nearest_node

    def steer(self, nearest_node, sampled_point):
        """
        This method should return a point in the viable set such that it is closer 
        to the nearest_node than sampled_point is.

        Args:
            nearest_node (Node): nearest node on the tree to the sampled point
            sampled_point (tuple of (float, float)): sampled point
        Returns:
            new_node (Node): new node created from steering
        """
        new_node = Tree_Node(nearest_node.x, nearest_node.y)
        sample_x, sample_y = sampled_point
        theta = math.atan2(sample_y - nearest_node.y, sample_x - nearest_node.x)
        new_node.x += self.expand_dis * math.cos(theta)
        new_node.y += self.expand_dis * math.sin(theta)
        new_node.parent = nearest_node
        return new_node
    
    def world_to_og_coords(self, x_w, y_w):
        """
        world coordinates:
            (At the start position and orientation of the car:)
            x-axis: Pointing the same as the front of the car
            y-axis: Pointing to the left of the car
        occupancy grid coordinates:
            (occupancy grid is 20*100(y*x) np array, resolution is 0.1)
            (to get point (x0,y0) in the occupancy grid, use occupancy_grid[y0,x0])
            (origin at right bottom corner of the occupancy grid, the car is always at the center of the occupancy grid(50,10) )
            x-axis: Pointing the same as the front of the car
            y-axis: Pointing to the left of the car
        """

        dist = math.sqrt(((self.posX - x_w)) ** 2 + 
                       ((self.posY - y_w)) ** 2) / self.resolution
        angle = math.atan2((self.posY - y_w), (self.posX - x_w))
        angle -= self.theta

        x_og = math.floor(self.occupancy_grid.shape[1] // 2 - dist * math.cos(angle))
        y_og = math.floor(self.occupancy_grid.shape[0] // 2 - dist * math.sin(angle))

        return x_og, y_og
    
    def og_coords_to_world(self, x_og, y_og):
        x_diff = x_og - self.occupancy_grid.shape[1] // 2
        y_diff = y_og - self.occupancy_grid.shape[0] // 2

        dist_x = x_diff * self.resolution
        dist_y = y_diff * self.resolution

        x_w = self.posX + (dist_x * math.cos(self.theta) + dist_y * math.sin(self.theta))
        y_w = self.posY + (dist_x * math.sin(self.theta) + dist_y * math.cos(self.theta))

        return x_w, y_w
    
    # def og_coords_to_world(self, x_og, y_og):
    #     # WRONG RN
    #     x_w = (x_og - self.occupancy_grid.shape[1] // 2) * self.resolution + self.posX
    #     y_w = (y_og - self.occupancy_grid.shape[0] // 2) * self.resolution + self.posY

    #     return x_w, y_w
    
    def bresenhams_algorithm(self, x0, y0, x1, y1):
        """
        Bresenham's line algorithm to generate points between (x0, y0) and (x1, y1).
        
        Args:
            x0, y0: Start point (int).
            x1, y1: End point (int).
            
        Returns:
            List of tuples: Points on the line.
        """
        points = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy  # Error value e_xy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy: 
                err += dy
                x0 += sx
            if e2 <= dx: 
                err += dx
                y0 += sy
        return points

    def check_collision(self, nearest_node, new_node):
        """
        This method should return whether the path between nearest and new_node is
        collision free.

        Args:
            nearest (Node): nearest node on the tree
            new_node (Node): new node from steering
        Returns:
            collision (bool): whether the path between the two nodes are in collision
                              with the occupancy grid
        """
        nearest_node_x, nearest_node_y = self.world_to_og_coords(nearest_node.x, nearest_node.y)
        new_node_x, new_node_y = self.world_to_og_coords(new_node.x, new_node.y)
        # print("nearest_node_x: ", nearest_node_x, "nearest_node_y: ", nearest_node_y)
        # print("new_node_x: ", new_node_x, "new_node_y: ", new_node_y)

        # Generate points(x,y) between the two nodes
        points = self.bresenhams_algorithm(nearest_node_x, nearest_node_y, new_node_x, new_node_y)
        points = np.array([np.array([pt[0], pt[1]]) for pt in points])
        
        if np.any(points < 0) or np.any(points[:, 0] >= self.occupancy_grid.shape[1]) or np.any(points[:, 1] >= self.occupancy_grid.shape[0]):
            return False # Out of Grid
        
        if np.any(self.occupancy_grid[points[:, 1], points[:, 0]] == 1):
            return False # Collision

        # for point in points:
        #     if 0 <= point[0] < self.occupancy_grid.shape[1] and 0 <= point[1] < self.occupancy_grid.shape[0] : # minus 1 to avoid out of index?
        #         # print("in our grid: ", point)
        #         if self.occupancy_grid[point[1], point[0]] == 1:
        #             # print("Collision at: ", point)
        #             # collision_point = self.og_coords_to_world(point[0], point[1])
        #             # self.visualize_one_point(self.og_coords_to_world(point[0], point[1]))
        #             # self.visualize_one_node(new_node)
        #             # time.sleep(5.0)
        #             return False # Collision！
        #     else:
        #         # pass
        #         # print("out of our grid:", point)
        #         return False # Out of our grid! Normally the new_node will not out of our grid

        return True # Collision Free

    def is_goal(self, latest_added_node, goal_x, goal_y):
        """
        This method should return whether the latest added node is close enough
        to the goal.

        Args:
            latest_added_node (Node): latest added node on the tree
            goal_x (double): x coordinate of the current goal
            goal_y (double): y coordinate of the current goal
        Returns:
            close_enough (bool): true if node is close enoughg to the goal
        """
        distance = np.linalg.norm(np.array([latest_added_node.x, latest_added_node.y]) - np.array([goal_x, goal_y]))
        
        return distance < self.is_goal_threshold

    def find_path(self, tree, latest_added_node):
        """
        This method returns a path as a list of Nodes connecting the starting point to
        the goal once the latest added node is close enough to the goal

        Args:
            tree ([]): current tree as a list of Nodes
            latest_added_node (Node): latest added node in the tree
        Returns:
            path ([]): valid path as a list of Nodes
        """
        n = latest_added_node
        path = []
        
        while n is not tree[0]:
            path.append(n)
            n = n.parent

        path.append(tree[0])
        
        return path[::-1]



    # The following methods are needed for RRT* and not RRT
    def cost(self, tree, node):
        """
        This method should return the cost of a node

        Args:
            node (Node): the current node the cost is calculated for
        Returns:
            cost (float): the cost value of the node
        """
        return 0

    def line_cost(self, n1, n2):
        """
        This method should return the cost of the straight line between n1 and n2

        Args:
            n1 (Node): node at one end of the straight line
            n2 (Node): node at the other end of the straint line
        Returns:
            cost (float): the cost value of the line
        """
        return 0

    def near(self, tree, node):
        """
        This method should return the neighborhood of nodes around the given node

        Args:
            tree ([]): current tree as a list of Nodes
            node (Node): current node we're finding neighbors for
        Returns:
            neighborhood ([]): neighborhood of nodes as a list of Nodes
        """
        neighborhood = []
        return neighborhood

def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    rrt_node = RRT()
    rclpy.spin(rrt_node)

    rrt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()