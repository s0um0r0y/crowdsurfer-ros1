#!/usr/bin/env python

import rospy
import numpy as np
import torch
import math as m
import tf2_ros
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Odometry

from models.vq_vae import VQVAE
from models.fused import FusedModel
from utils.trajectory import visualise_trajectory


class OpenLoopBag:
    def __init__(self):
        rospy.init_node('open_loop_bag', anonymous=True)

        # Initialize and load models
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vqvae = VQVAE().to(self.device)
        self.vqvae.load_state_dict(torch.load('/home/soumoroy/Downloads/crowd_surfer-master/checkpoints/state_dict/vqvae.pth',map_location=self.device))
        self.vqvae.eval()

        self.pixelcnn = FusedModel().to(self.device)
        self.pixelcnn.load_state_dict(torch.load('/home/soumoroy/Downloads/crowd_surfer-master/checkpoints/state_dict/pixelcnn.pth',map_location=self.device))
        self.pixelcnn.eval()

        # Subscribers
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.laser_scan_callback)
        self.marker_sub = rospy.Subscriber('/marker', MarkerArray, self.marker_callback)
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)

        # Publishers
        self.grid_map_pub = rospy.Publisher('/grid_map', OccupancyGrid, queue_size=10)
        self.trajectory_pub = rospy.Publisher('/pred_trajectory', Path, queue_size=10)
        self.sampled_trajectory_publisher_1 = rospy.Publisher('/sampled_trajectory_1', Path)
        self.sampled_trajectory_publisher_2 = rospy.Publisher('/sampled_trajectory_2', Path)
        self.sampled_trajectory_publisher_3 = rospy.Publisher('/sampled_trajectory_3', Path)
        self.sampled_trajectory_publisher_4 = rospy.Publisher('/sampled_trajectory_4', Path)
        self.sampled_trajectory_publisher_5 = rospy.Publisher('/sampled_trajectory_5', Path)
        self.sampled_trajectory_publishers = [self.sampled_trajectory_publisher_1, self.sampled_trajectory_publisher_2, self.sampled_trajectory_publisher_3, self.sampled_trajectory_publisher_4, self.sampled_trajectory_publisher_5]
   
        # self.dynamic_obstacles = np.zeros((5, 4, 10))
        # self.heading_to_goal = np.zeros(2)
        self.max_obstacles = 10  # Maximum number of obstacles to track
        self.num_timesteps = 5   # Number of timesteps for dynamic obstacles
        # self.data_format = (self.num_timesteps, 4, self.max_obstacles)
        # self.dynamic_obstacles = np.full(self.data_format, np.nan, dtype=np.float32)
        self.heading = np.zeros([1])
        self.marker_positions = {}  # To store previous positions and timestamps

        # Timer for inference
        # rospy.Timer(rospy.Duration(0.5), self.timer_callback)
        # self.timer = self.create_timer(0.5, self.timer_callback)

        while not rospy.is_shutdown():
            self.infer_trajectories()

    def laser_scan_to_grid(self, scan, grid_size=60, resolution=0.1, max_range=30.0):
        """
        Convert LaserScan data to an occupancy grid map.
        """
        grid = -1 * np.ones((grid_size, grid_size), dtype=np.int8)
        center = grid_size // 2

        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))

        for r, theta in zip(scan.ranges, angles):
            if 0 < r < max_range:
                x = int(center + (r * np.cos(theta)) / resolution)
                y = int(center + (r * np.sin(theta)) / resolution)
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    grid[y, x] = 100

        self.occupancy_grid = grid
        # Convert to OccupancyGrid message
        occupancy_grid_msg = OccupancyGrid()
        occupancy_grid_msg.header=scan.header
        # occupancy_grid_msg.header.stamp = rospy.Time.now()
        # occupancy_grid_msg.header.frame_id = "map"
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
        occupancy_grid_msg = self.laser_scan_to_grid(scan=msg,grid_size=60, resolution=0.1, max_range=msg.range_max)
        self.grid_map_pub.publish(occupancy_grid_msg)

    def marker_callback(self, msg):
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
        
        if not hasattr(self, 'heading'):
            rospy.logwarn("No heading available for inference.")
            return

        # Prepare input tensors
        occupancy_grid = torch.tensor(self.occupancy_grid).unsqueeze(0).unsqueeze(0).float().to(self.device)
        dynamic_obstacles = torch.tensor(self.dynamic_obstacles).unsqueeze(0).float().to(self.device)
        heading = torch.tensor(self.heading).unsqueeze(0).float().to(self.device)

        assert occupancy_grid.shape == (1, 1, 60, 60), f"Expected shape [1, 1, 60, 60], got {occupancy_grid.shape}"
        assert dynamic_obstacles.shape == (1, 5, 4, 10), f'Expected shape [1, 5, 4, 10] got {dynamic_obstacles.shape}'
        assert heading.shape == (1, 1), f'Expected shape [1, 1] got {heading.shape}'
        
        pixelcnn_embedding = self.pixelcnn(occupancy_grid, dynamic_obstacles, heading).permute(0, 2, 1)

        _, pixelcnn_idx = torch.max(pixelcnn_embedding, dim=1)
        
        pred_traj_coeffs = self.vqvae.from_indices(pixelcnn_idx).view(2, 11)
        
        # Publish predicted trajectory
        self.publish_trajectory(pred_traj_coeffs, self.trajectory_pub)

        pixelcnn_idx = torch.multinomial(torch.nn.functional.softmax(pixelcnn_embedding.squeeze().permute(1, 0)), 5).permute(1, 0)
        for i in range(5):
            pred_traj = self.vqvae.from_indices(pixelcnn_idx[i].unsqueeze(0)).view(2, 11)
            self.publish_trajectory(pred_traj, self.sampled_trajectory_publishers[i])
        return

    def publish_trajectory(self, trajectory_coeffs, publisher):
        """
        Publish a trajectory as a Path message.
        
        Args:
            trajectory_coeffs: Tensor of shape [2, 11].
            publisher: ROS Publisher object.
        """
        coefficients = trajectory_coeffs.cpu().detach().numpy() # Ensure it's on CPU
        coefficients_x = coefficients[0, :]  # First row -> X coefficients
        coefficients_y = coefficients[1, :]  # Second row -> Y coefficients

        # Compute trajectory points
        X, Y = visualise_trajectory(coefficients_x, coefficients_y)

        # Create Path message
        path_msg = Path()       
        path_msg.header.frame_id = "base_link"  # Set to appropriate frame
        path_msg.header.stamp = rospy.Time.now()

        for x, y in zip(X, Y):
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
