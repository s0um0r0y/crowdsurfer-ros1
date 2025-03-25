#!/usr/bin/env python

import rospy
import rosbag
import numpy as np
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
import torch

class RosInterfacePixelCNN:
    def __init__(self):
        rospy.init_node('ros_interface_pixelcnn', anonymous=True)
        self.occupancy_grid = np.zeros((1, 60, 60))
        self.dynamic_obstacles = np.zeros((5, 4, 10))
        self.heading_to_goal = np.zeros(2)

        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.heading_to_goal_sub = rospy.Subscriber('/odom', Odometry, self.heading_to_goal_callback)
        self.marker_sub = rospy.Subscriber('/marker', MarkerArray, self.marker_callback)

        self.grid_size = 60  # Grid dimensions (60x60)
        self.grid_resolution = 0.1  # Resolution of each cell in meters
        self.grid_center = self.grid_size // 2  # Center of the grid

        self.max_obstacles = 10  # Maximum number of obstacles to track
        self.num_timesteps = 5   # Number of timesteps for dynamic obstacles
        self.data_format = (self.num_timesteps, 4, self.max_obstacles)

        # Initialize storage for obstacle data
        self.dynamic_obstacles = np.full(self.data_format, np.nan, dtype=np.float32)
        self.previous_data = {}  # To store previous positions and timestamps

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        range_max = msg.range_max

        # Convert laser scan ranges to grid coordinates
        for i, r in enumerate(ranges):
            if r < range_max:  # Ignore invalid or out-of-range values
                angle = angle_min + i * angle_increment
                x = int(self.grid_center + r * np.cos(angle) / self.grid_resolution)
                y = int(self.grid_center + r * np.sin(angle) / self.grid_resolution)

                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    self.occupancy_grid[0, y, x] = 100  # Mark cell as occupied

        rospy.loginfo("Updated Occupancy Grid Map")

        # print("OGM : ",self.occupancy_grid)
    
    def heading_to_goal_callback(self, msg):
        # Extract position and orientation from Odometry message
        position_x = msg.pose.pose.position.x
        position_y = msg.pose.pose.position.y
        orientation_z = msg.pose.pose.orientation.z  # Quaternion z
        orientation_w = msg.pose.pose.orientation.w  # Quaternion w

        # Calculate heading angle (yaw) using quaternion components
        heading_angle = 2 * np.arctan2(orientation_z, orientation_w)

        # Update heading-to-goal array
        self.heading_to_goal[0] = heading_angle  # Heading angle in radians
        self.heading_to_goal[1] = np.linalg.norm(position_x, position_y)  # Distance to goal (example placeholder)

        rospy.loginfo(f"Heading to Goal: {self.heading_to_goal}")
        # print("dyanamic obstacle shape : ",self.heading_to_goal.shape)

    def marker_callback(self, msg):
        current_timestamp = rospy.Time.now().to_sec()  # Get current time in seconds
        obstacle_positions = []

        # Extract positions from Marker message (assuming `msg.points` contains obstacle positions)
        for marker in msg.markers[:self.max_obstacles]:  # Limit to max_obstacles
            position = marker.pose.position
            obstacle_positions.append([position.x, position.y])

        # Convert positions to numpy array
        obstacle_positions = np.array(obstacle_positions)

        # Initialize velocity array
        velocities = np.zeros_like(obstacle_positions)

        # Calculate velocities based on previous data
        for i, position in enumerate(obstacle_positions):
            obstacle_id = i  # Use index as a simple ID (replace with actual ID if available)
            if obstacle_id in self.previous_data:
                prev_timestamp, prev_position = self.previous_data[obstacle_id]
                dt = current_timestamp - prev_timestamp

                if dt > 0:  # Avoid division by zero
                    velocities[i] = (position - prev_position) / dt

            # Update previous data with current position and timestamp
            self.previous_data[obstacle_id] = (current_timestamp, position)

        # Combine positions and velocities into a single array [x, y, vx, vy]
        obstacle_data = np.hstack((obstacle_positions, velocities))

        # Update dynamic_obstacles array with new data
        obstacle_data_transposed = obstacle_data.T  # Shape: (4, num_obstacles)
        self.dynamic_obstacles[:-1] = self.dynamic_obstacles[1:]  # Shift timesteps
        padded_data = np.pad(obstacle_data_transposed,
        ((0, 0), (0, max(0, self.max_obstacles - obstacle_data_transposed.shape[1]))),  # Pad if fewer obstacles
        mode='constant',
        constant_values=np.nan,
        )
        self.dynamic_obstacles[-1] = padded_data
        rospy.loginfo("Dynamic Obstacle Processor Node Running...")

        # print("dyanamic obstacle : ",self.dynamic_obstacles)

    # def save_to_npy(self):
    #     np.save('occupancy_grid.npy', self.occupancy_grid)
    #     np.save('dynamic_obstacles.npy', self.dynamic_obstacles)
    #     np.save('heading_to_goal.npy', self.heading_to_goal)

    def run(self):
        rospy.spin()
        # self.save_to_npy()

if __name__ == '__main__':
    node = RosInterfacePixelCNN()
    node.run()

