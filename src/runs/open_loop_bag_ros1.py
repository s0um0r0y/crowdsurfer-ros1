#!/usr/bin/env python3

import rospy
import numpy as np
import torch
import math as m
import tf2_ros
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav_msgs.msg import Odometry

from models.vq_vae import VQVAE
from models.fused import FusedModel
from utils.trajectory import visualise_trajectory
import mpc_non_dy
from jax import random
import jax.numpy as jnp
import time
import projection_guidance
from dataclasses import dataclass
from torch import Tensor
from typing import Optional

from priest_core import State, Obstacles, Priest
from crowd_surfer_priest import Planner

@dataclass
class InferenceData:
    """
    Dataclass to hold the input data for the pipeline. All data is in Ego frame.

    Attributes:
        static_obstacles: Tensor of shape (batch_size, 1, height, width) or (batch_size, 2, max_points)
            if static_obstacle_type is OCCUPANCY_MAP [DEFAULT] or DOWNSAMPLED_POINT_CLOUD
        dynamic_obstacles: Tensor of shape (batch_size, num_previous_timesteps, 4, max_obstacles)
            where num_previous_timesteps is 5 by default, and 4 channels corresponds to [x, y, vx, vy]
        heading_to_goal: Tensor of shape (batch_size, 1)
            where 1 corresponds to the heading angle in radians
        ego_velocity_for_projection: Tensor of shape (batch_size, 2)
            where 2 corresponds to [vx, vy]
        goal_position_for_projection: Tensor of shape (batch_size, 2)
            where 2 corresponds to [x, y]
        obstacle_positions_for_projection: Tensor of shape (batch_size, 2, max_projection_dynamic_obstacles + max_projection_static_obstacles)
        obstacle_velocities_for_projection: Tensor of shape (batch_size, 2, max_projection_dynamic_obstacles + max_projection_static_obstacles)
            where max_projection_dynamic_obstacles and max_projection_static_obstacles are 10 and 50 by default respectively.
    """

    static_obstacles: Tensor
    dynamic_obstacles: Tensor
    heading_to_goal: Tensor
    ego_velocity_for_projection: Tensor
    goal_position_for_projection: Tensor
    obstacle_positions_for_projection: Tensor
    obstacle_velocities_for_projection: Tensor
    ego_acceleration_for_projection: Tensor = None

    def __post_init__(self):
        if self.ego_acceleration_for_projection is None:
            self.ego_acceleration_for_projection = torch.zeros_like(
                self.ego_velocity_for_projection
            )

@dataclass
class PlanningData:
    """
    Dataclass to store the data required for planning
    All values are in ego frame, except odometry which is in world frame
    Dynamic obstacles include the past 5 timesteps of dynamic obstacles
    """

    goal: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    point_cloud: np.ndarray
    laser_scan: np.ndarray
    dynamic_obstacles: np.ndarray
    update_waypoints: bool = False  # True
    sub_goal: Optional[np.ndarray] = None
    # global_path: Optional[np.ndarray] = None

class OpenLoopBag:
    def __init__(self):
        rospy.init_node('open_loop_bag', anonymous=True)

        # initialize and load the planner
        self.planner = Planner()

        # Initialize and load models
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vqvae = VQVAE().to(self.device)
        self.vqvae.load_state_dict(torch.load('/home/soumoroy/Downloads/crowd_surfer-master/checkpoints/state_dict/vqvae.pth',
                                              map_location=self.device))
        self.vqvae.eval()

        self.pixelcnn = FusedModel().to(self.device)
        self.pixelcnn.load_state_dict(torch.load('/home/soumoroy/Downloads/crowd_surfer-master/checkpoints/state_dict/pixelcnn.pth',
                                                 map_location=self.device))
        self.pixelcnn.eval()

        # Subscribers
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.laser_scan_callback)
        self.marker_sub = rospy.Subscriber('/marker', MarkerArray, self.marker_callback)
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)

        # VQ-VAE+ PixelCNN generated trajectory Publishers
        # static and dynamic obstacles
        self.grid_map_pub = rospy.Publisher('/grid_map', OccupancyGrid, queue_size=10)
        self.trajectory_pub = rospy.Publisher('/pred_trajectory', Path, queue_size=10)
        self.sampled_trajectory_publisher_1 = rospy.Publisher('/sampled_trajectory_1', Path, queue_size=10)
        self.sampled_trajectory_publisher_2 = rospy.Publisher('/sampled_trajectory_2', Path, queue_size=10) 
        self.sampled_trajectory_publisher_3 = rospy.Publisher('/sampled_trajectory_3', Path, queue_size=10)
        self.sampled_trajectory_publisher_4 = rospy.Publisher('/sampled_trajectory_4', Path, queue_size=10)
        self.sampled_trajectory_publisher_5 = rospy.Publisher('/sampled_trajectory_5', Path, queue_size=10)
        self.sampled_trajectory_publishers = [self.sampled_trajectory_publisher_1, 
                                              self.sampled_trajectory_publisher_2, 
                                              self.sampled_trajectory_publisher_3, 
                                              self.sampled_trajectory_publisher_4, 
                                              self.sampled_trajectory_publisher_5]
        
        # for PRIEST-optimized trajectories publishers
        self.optimized_trajectory_publisher_1 = rospy.Publisher('/optimized_trajectory_1', Path, queue_size=10)
        self.optimized_trajectory_publisher_2 = rospy.Publisher('/optimized_trajectory_2', Path, queue_size=10)
        self.optimized_trajectory_publisher_3 = rospy.Publisher('/optimized_trajectory_3', Path, queue_size=10)
        self.optimized_trajectory_publisher_4 = rospy.Publisher('/optimized_trajectory_4', Path, queue_size=10)
        self.optimized_trajectory_publisher_5 = rospy.Publisher('/optimized_trajectory_5', Path, queue_size=10)
        self.optimized_trajectory_publishers = [self.optimized_trajectory_publisher_1,
                                                self.optimized_trajectory_publisher_2,
                                                self.optimized_trajectory_publisher_3,
                                                self.optimized_trajectory_publisher_4,
                                                self.optimized_trajectory_publisher_5]
   
        # self.optimized_trajectory_publishers = rospy.Publisher('/optimized_trajectory', Path)

        # self.dynamic_obstacles = np.zeros((5, 4, 10))
        # self.heading_to_goal = np.zeros(2)
        self.max_obstacles = 10  # Maximum number of obstacles to track
        self.num_timesteps = 5   # Number of timesteps for dynamic obstacles
        # self.data_format = (self.num_timesteps, 4, self.max_obstacles)
        # self.dynamic_obstacles = np.full(self.data_format, np.nan, dtype=np.float32)
        self.heading = np.zeros([1])
        self.marker_positions = {}  # To store previous positions and timestamps
        self.x_init = 1.0
        self.y_init = 2
        self.x_fin = 2.0
        self.y_fin = 15

        self.vx_init = 0.05
        self.vy_init = 0.1
        self.ax_init = 0.0
        self.ay_init = 0.0
        self.vx_fin = 0.0
        self.vy_fin = 0.0
        self.ax_fin = 0.0
        self.ay_fin = 0.0
        self.v_des = 1.0

        # Timer for inference
        # rospy.Timer(rospy.Duration(0.5), self.timer_callback)
        # self.timer = self.create_timer(0.5, self.timer_callback)

        while not rospy.is_shutdown():
            self.infer_trajectories()

    def laser_scan_to_grid(self, scan, grid_size=60, resolution=0.1, max_range=30.0):
        """
        Convert LaserScan data to an occupancy grid map.
        """
        self.static_obstacles = []
        grid = -1 * np.ones((grid_size, grid_size), dtype=np.int8)
        center = grid_size // 2

        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))

        for r, theta in zip(scan.ranges, angles):
            if 0 < r < max_range:
                x = int(center + (r * np.cos(theta)) / resolution)
                y = int(center + (r * np.sin(theta)) / resolution)
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    grid[y, x] = 100
                    self.static_obstacles.append([x, y])

        self.static_obstacles = np.asarray(self.static_obstacles)

        self.occupancy_grid = grid

        # Convert to OccupancyGrid message
        occupancy_grid_msg = OccupancyGrid()
        occupancy_grid_msg.header=scan.header
        occupancy_grid_msg.info.resolution = resolution
        occupancy_grid_msg.info.width = grid_size
        occupancy_grid_msg.info.height = grid_size
        occupancy_grid_msg.info.origin = Pose()
        occupancy_grid_msg.info.origin.position.x = -grid_size * resolution / 2
        occupancy_grid_msg.info.origin.position.y = -grid_size * resolution / 2
        occupancy_grid_msg.data = grid.flatten().tolist()

        return occupancy_grid_msg

    def laser_scan_callback(self, msg):
        """
        Callback for LaserScan messages.
        """
        self.time = msg.header.stamp
        occupancy_grid_msg = self.laser_scan_to_grid(scan=msg,grid_size=60, resolution=0.1, max_range=msg.range_max)
        self.grid_map_pub.publish(occupancy_grid_msg)

    # this function needs to be edited
    def marker_callback(self, msg):
        """
        Callback for Marker messages.
        """

        # read and store marker data
        num_markers = len(msg.markers)
        for i in range(num_markers):
            marker = msg.markers[i]
            time = marker.header.stamp
            id = marker.id
            x = marker.pose.position.x
            y = marker.pose.position.y

            if str(id) in self.marker_positions.keys():
                marker_data = self.marker_positions[str(id)]
                prev_time = marker_data[-1][0]
                prev_x = marker_data[-1][1]
                prev_y = marker_data[-1][2]
                delta_t = time - prev_time
                delta_t = delta_t.secs + delta_t.nsecs * 1e-9
                u = (x - prev_x) / delta_t
                v = (y - prev_y) / delta_t
                if len(marker_data) < 5:
                    marker_data.append((time, x, y, u, v))
                if len(marker_data) == 5:
                    marker_data.pop(0)
                    marker_data.append((time, x, y, u, v))
            else:
                self.marker_positions[str(id)] = [(time, x, y, None, None)]

        # process the stored marker data and get the markers closest to the robot
        distances = {}
        for i in self.marker_positions.keys():
            distances[i] = m.sqrt(self.marker_positions[i][-1][1] ** 2 + self.marker_positions[i][-1][2] ** 2)
        distances = torch.tensor(list(distances.values()))
        _, idx = torch.topk(distances, k=10, largest=False)
        dynamic_obstacles = []
        for i in idx:
            dynamic_obstacles.append([data[1:] for data in self.marker_positions[str(i.item())]])
        self.dynamic_obstacles = torch.tensor(dynamic_obstacles).permute(1,2,0)
        return

    def infer_trajectories(self):
        """
        Infer trajectories using VQVAE and PixelCNN models.
        """
        if not hasattr(self, 'occupancy_grid'):
            rospy.logwarn("No Occupancy Grid available for inference.")
            return
        
        if not hasattr(self, 'dynamic_obstacles'):
            rospy.logwarn("No Dynamic available for inference.")
            return
        
        # if not hasattr(self, 'heading'):
        #     rospy.logwarn("No heading available for inference.")
        #     return

        state_initial = State(0, 0, 0.1, 0, 0, 0, 0, 0)
        state_goal = State(3, 3, 3, 3, 3, 3, 3, 3)

        # Prepare input tensors
        try:
            obstacles = Obstacles(self.static_obstacles[:, 0], 
                                  self.static_obstacles[:, 1], 
                                  self.dynamic_obstacles[4, 0, :].numpy(), 
                                  self.dynamic_obstacles[4, 1, :].numpy(), 
                                  self.dynamic_obstacles[4, 2, :].numpy(), 
                                  self.dynamic_obstacles[4, 3, :].numpy())
            
        except:
            rospy.logerr("Slicing error in obstacle dataclass.")
            print(self.static_obstacles[:, 0])
            print(self.static_obstacles[:, 1])
            print(self.dynamic_obstacles[4, 0, :].numpy())
            print(self.dynamic_obstacles[4, 1, :].numpy())
            print(self.dynamic_obstacles[4, 2, :].numpy())
            print(self.dynamic_obstacles[4, 3, :].numpy())
            return 0, 0, 0, 0
        
        c_x, c_y, x, y, x_vqvae, y_vqvae ,vx_control, vy_control, ax_control, ay_control, norm_v_t, angle_v_t, obstacles_dict= self.planner.generate_trajectory(self.occupancy_grid, 
                                                                                                                                                                state_initial,                                                                                                                        
                                                                                                                                                                state_goal, 
                                                                                                                                                                obstacles)

        x = np.asarray(x)
        print("the x array o/p : ",x)
        y = np.asarray(y)
        x_vqvae = np.asarray(x_vqvae)
        y_vqvae = np.asarray(y_vqvae)

        occupancy_grid = torch.tensor(self.occupancy_grid).unsqueeze(0).unsqueeze(0).float().to(self.device)
        dynamic_obstacles = torch.tensor(self.dynamic_obstacles).unsqueeze(0).float().to(self.device)
        heading = torch.tensor(self.heading).unsqueeze(0).float().to(self.device)

        assert occupancy_grid.shape == (1, 1, 60, 60), f"Expected shape [1, 1, 60, 60], got {occupancy_grid.shape}"
        assert dynamic_obstacles.shape == (1, 5, 4, 10), f'Expected shape [1, 5, 4, 10] got {dynamic_obstacles.shape}'
        assert heading.shape == (1, 1), f'Expected shape [1, 1] got {heading.shape}'
        
        # Output shape is likely [batch, latent_dim, seq_len], so .permute(0, 2, 1) → [batch, seq_len, latent_dim]
        pixelcnn_embedding = self.pixelcnn(occupancy_grid, dynamic_obstacles, heading).permute(0, 2, 1)

        # Gets the argmax over latent classes (i.e., most likely codebook index) for each timestep in the sequence
        # pixelcnn_idx has shape [batch, seq_len] — a discrete sequence of indices.
        _, pixelcnn_idx = torch.max(pixelcnn_embedding, dim=1)
        
        # Passes indices into the VQ-VAE decoder to get continuous trajectory coefficients.
        # 2D positions × 11 timesteps
        pred_traj_coeffs = self.vqvae.from_indices(pixelcnn_idx).view(2, 11)
        # print("predicted trajectory coeffs", pred_traj_coeffs)
        
        # Publish predicted trajectory
        # self.publish_trajectory(pred_traj_coeffs, self.trajectory_pub)

        # pixelcnn_embedding.squeeze() removes batch dim: [seq_len, latent_dim]
        # softmax(..., dim=1): converts logits to probabilities over latent codes.
        # torch.multinomial(..., 5): samples 5 different latent sequences from the probability distribution.
        # .permute(1, 0): brings it back to [5, seq_len] (5 different index sequences).
        pixelcnn_idx = torch.multinomial(torch.nn.functional.softmax(pixelcnn_embedding.squeeze().permute(1, 0)), 5).permute(1, 0)

        # # Creates a list of Twist messages (vel_cmds) to store control commands for obstacles.
        # vel_cmds = [Twist() for _ in range(10)]
        # vel_cmd = Twist()

        for i in range(5):
        # For each of the 5 sampled discrete trajectories:
        # Decodes it using VQ-VAE.
        # Reshapes and publishes it using a separate trajectory publisher.
            # pred_traj = self.vqvae.from_indices(pixelcnn_idx[i].unsqueeze(0)).view(2, 11)
            self.publish_trajectory(x_vqvae[i, :], y_vqvae[i, :], self.sampled_trajectory_publishers[i])
            self.publish_trajectory(x, y, self.optimized_trajectory_publishers[i])

            # Convert the trajectory to the expected format for input to PRIEST
            # samples = draw_gaussian_samples(mean, cov) -> pixelcnn_embedding
            # sample more coefficients and get new coefficients from PRIEST

            # init_traj = self.vqvae.from_indices(pixelcnn_idx[i].unsqueeze(0)).view(2, 11)

            # a_obs_1, b_obs_1, a_obs_2, b_obs_2 = self.get_obs_trajectories()

        return

    def publish_trajectory(self, x, y, publisher):
        """
        Publish a trajectory as a Path message.
        
        Args:
            trajectory_coeffs: Tensor of shape [2, 11].
            publisher: ROS Publisher object.
        """

        # Create Path message
        path_msg = Path()       
        path_msg.header.frame_id = "base_link"  # Set to appropriate frame
        path_msg.header.stamp = rospy.Time.now()

        for x, y in zip(x, y):
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = path_msg.header.stamp
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0  # Assume 2D trajectory

            path_msg.poses.append(pose)

        publisher.publish(path_msg)
        # rospy.loginfo(f"Published trajectory with {len(path_msg.poses)} points.")


    def run(self):
        """
        Run the ROS node.
        """
        rospy.spin()

if __name__ == '__main__':
    try:
        node = OpenLoopBag()
        node.run()
    except rospy.ROSInterruptException:
        pass
