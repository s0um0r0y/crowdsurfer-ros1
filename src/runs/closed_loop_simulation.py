import os
import rospy
import tf2_ros

from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav_msgs.msg import Path
from pedsim_msgs.msg import TrackedPersons
from tf2_geometry_msgs import do_transform_pose

import torch
import math as m
import numpy as np

from priest_core import State, Obstacles, Priest
from crowd_surfer_priest import Planner
from utils.trajectory import visualise_trajectory

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

class ClosedLoopSimulation():
    def __init__(self):
        rospy.init_node('closed_loop_sim', anonymous=True)

        # init and load the planner
        self.planner = Planner()

        # subscriber 
        rospy.Subscriber('/scan', LaserScan, self.laser_scan_callback)
        rospy.Subscriber('/pedsim_visualizer/tracked_persons', TrackedPersons, self.marker_callback)
        rospy.Subscriber('/crowsurfer_goal',PoseStamped, self.goal_callback)
        rospy.Subscriber('/odom', Odometry, self.update_odometry)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # publisher
        self.static_obstacles_publisher = rospy.Publisher('/static_obs', MarkerArray)
        self.dynamic_obstacles_publisher = rospy.Publisher('/dynamic_obs', MarkerArray)
        self.occupancy_grid_publisher = rospy.Publisher('/grid_map', OccupancyGrid)
        self.trajectory_publisher = rospy.Publisher('pred_trajectory', Path, 10)
        self.sampled_trajectory_publisher_1 = rospy.Publisher('/sampled_trajectory_1', Path)
        self.sampled_trajectory_publisher_2 = rospy.Publisher('/sampled_trajectory_2', Path)
        self.sampled_trajectory_publisher_3 = rospy.Publisher('/sampled_trajectory_3', Path)
        self.sampled_trajectory_publisher_4 = rospy.Publisher('/sampled_trajectory_4', Path)
        self.sampled_trajectory_publisher_5 = rospy.Publisher('/sampled_trajectory_5', Path)

        self.optimized_trajectory_publisher = rospy.Publisher('/optimized_trajectory', Path)

        self.cmd_vel_publisher = rospy.Publisher('/optimized_trajectory',Path)
        self.sampled_trajectory_publishers = [self.sampled_trajectory_publisher_1,
                                              self.sampled_trajectory_publisher_2,
                                              self.sampled_trajectory_publisher_3,
                                              self.sampled_trajectory_publisher_4,
                                              self.sampled_trajectory_publisher_5]
        
        self.marker_positions = {}

        self.goal_reached = False
        while not rospy.is_shutdown():
            self.plan()

    def update_odometry(self, odom):
        self.current_x = odom.pose.pose.position.x
        self.current_y = odom.pose.pose.position.y
        self.current_vx = odom.twist.twist.linear.x
        self.current_vy = odom.twist.twist.linear.y

    def laser_scan_to_grid(self, scan , grid_size = 60, resolution = 0.1, max_range = 30.0):
        self.static_obstacles = []
        grid = -1 * np.ones((grid_size, grid_size), dtype=np.int8)
        center = grid_size // 2

        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))

        for r, theta in zip(scan.ranges, angles):
            if 0 < r < max_range:
                x = int(center + (r * np.cos(theta)) / resolution)
                y = int(center + (r * np.sin(theta)) / resolution)
                if 0 <= x < grid_size and 0 <= grid_size:
                    grid[y, x] = 100
                    self.static_obstacles.append([r*np.cos(theta), r*np.sin(theta)])

        self.static_obstacles = np.asarray(self.static_obstacles)
        N = self.static_obstacles.shape[0]
        self.static_obstacles = np.pad(self.static_obstacles,
                                       pad_width=((0, max(0, 100 - N)), (0, 0)),
                                       mode='constant',
                                       constant_values=0)
        
        self.occupancy_grid = grid

        # Convert to occupancy grid message
        occupancy_grid = OccupancyGrid()
        occupancy_grid.header = scan.header
        occupancy_grid.info.resolution = resolution
        occupancy_grid.info.width = grid_size
        occupancy_grid.info.height = grid_size
        occupancy_grid.info.origin = Pose()
        occupancy_grid.info.origin.x = -grid_size * resolution / 2
        occupancy_grid.info.origin.y = -grid_size * resolution / 2
        occupancy_grid.data = grid.flatten().tolist()

        return occupancy_grid
    
    def laser_scan_callback(self, msg_data):
        self.time = msg_data.header.stamp
        occupancy_grid = self.laser_scan_to_grid(scan=msg_data, grid_size=60, resolution=0.1, max_range=msg_data.range_max)
        self.occupancy_grid_publisher.publish(occupancy_grid)

    def marker_callback(self, msg_data):

        # read and store marker data
        num_markers = len(msg_data.tracks)
        for i in range(num_markers):
            marker = msg_data.tracks[i]
            id = marker.track_id
            x = marker.pose.pose.position.x
            y = marker.pose.pose.position.y
            vx = marker.twist.twist.linear.x
            vy = marker.twist.twist.linear.y

            if str(id) in self.marker_positions.keys():
                marker_data = self.marker_positions[str(id)]
                u = vx
                v = vy
                if len(marker_data) < 5:
                    marker_data.append((0, x, y, u, v))
                if len(marker_data) == 5:
                    marker_data.pop(0)
                    marker_data.append((0, x, y, u, v))

            else:
                self.marker_positions[str(id)] = [(0, x, y, None, None)]

        # process stord marker data and get the marker closest to the robot
        distances = {}
        for i in self.marker_positions.keys():
            distances[i] = m.sqrt(self.marker_positions[i][-1][1]**2 + self.marker_positions[i][-1][2]**2)
        distances = torch.tensor(list(distances.values()))
        _, idx = torch.topk(distances, k=10, largest=False)
        dynamic_obstacles = []
        for i in idx:
            dynamic_obstacles.append([data[1:] for data in self.marker_positions[str(i.item()+1)]])
        self.dynamic_obstacles = torch.tensor(dynamic_obstacles).permute(1, 2, 0)
        return
    
    def transform_base_link_to_map(self, position):
        base_link_position = PoseStamped()
        base_link_position.header.frame_id = "base_link"
        base_link_position.header.stamp = rospy.TIme.now()
        base_link_position.pose.position.x = position[0]
        base_link_position.pose.position.y = position[1]
        base_link_position.pose.position.z = 0

        transform = self.tf_buffer.lookup_transform("map", "base_link", rospy.TIme(0), rospy.Duration(1.0))
        map_position = do_transform_pose(base_link_position, transform)
        x = map_position.pose.position.x
        y = map_position.pose.position.y
        z = map_position.pose.position.z
        return x, y, z
    
    def create_marker(self, marker_id, position, color, scale=0.2, frame_id="base_link"):
        x, y, z =self.transform_base_link_to_map(position)
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "marker_array"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale

        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0

        marker.lifetime = rospy.Duration()

        return marker
    
    def visualise_obstacles(self, obstacles_dict):
        static_x = obstacles_dict["static_x"]
        static_y = obstacles_dict["static_y"]
        dynamic_x = obstacles_dict["dynamic_x"]
        dynamic_y = obstacles_dict["dynamic_y"]

        static_x = static_x[:, 0]
        static_y = static_y[:, 0]

        dynamic_x = dynamic_x[:, 0]
        dynamic_y = dynamic_y[:, 0]

        static_marker_array = MarkerArray()

        # publish markers for static_obstacles
        for i, (x, y) in enumerate(zip(static_x, static_y)):
            static_marker_array.markers.append(self.create_marker(i, (x, y, 0), (0.0, 0.5, 1.0)))
        self.static_obstacles_publisher.publish(static_marker_array)

        dynamic_marker_array = MarkerArray()

        # publish markers for dynamic_obstacles
        for i, (x, y) in enumerate(zip(dynamic_x, dynamic_y)):
            dynamic_marker_array.markers.append(self.create_marker(i, (x, y, 0), (0.0, 0.5, 1.0)))
        self.dynamic_obstacles_publisher.publish(dynamic_marker_array)

        return
    
    def infer_trajectories(self, state_initial, state_final):

        if not hasattr(self, 'occupancy_grid'):
            rospy.logerr("Not recieved  any occupancy grid map")
            return 0, 0, 0, 0
        if not hasattr(self, 'dynamic_obstacles'):
            rospy.logerr("Not recieved dynamic obstacles")
            return 0, 0, 0, 0
        
        try:
            obstacles = Obstacles(self.static_obstacles[:, 0], 
                                  self.static_obstacles[:, 1], 
                                  self.dynamic_obstacles[4, 0, :].numpy(), 
                                  self.dynamic_obstacles[4, 1, :].numpy(), 
                                  self.dynamic_obstacles[4, 2, :].numpy(), 
                                  self.dynamic_obstacles[4, 3, :].numpy())
        except:
            rospy.logerr("Slicing error in obstacles ")
            print(self.static_obstacles[:, 0])
            print(self.static_obstacles[:, 1])
            print(self.dynamic_obstacles[4, 0, :].numpy())
            print(self.dynamic_obstacles[4, 1, :].numpy())
            print(self.dynamic_obstacles[4, 2, :].numpy())
            print(self.dynamic_obstacles[4, 3, :].numpy())
            return 0,0,0,0
        
        c_x, c_y, x, y, x_vqvae, y_vqvae, vx_control, vy_control, ax_control, ay_control, norm_v_t, angle_v_t, obstacles_dict = self.planner.generate_trajectory(self.occupancy_grid, 
        state_initial, 
        state_final, 
        obstacles)

        self.visualise_obstacles(obstacles_dict)

        x = np.asarray(x)
        y = np.asarray(y)
        x_vqvae = np.asarray(x_vqvae)
        y_vqvae = np.asarray(y_vqvae)

        for i in range(5):
            self.publish_trajectory(x_vqvae[i, :], y_vqvae[i, :], self.sampled_trajectory_publishers[i])

        self.publish_trajectory(x, y, self.optimized_trajectory_publisher)

        return c_x, c_y,  norm_v_t, angle_v_t
    
    def publish_trajectory(self, x, y, publisher):

        # create path message 
        path_msg = Path()
        path_msg.header.frame_id = "base_link"
        path_msg.header.stamp = rospy.Time.now()

        for x, y in zip(x, y):
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = path_msg.header.stamp
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0

            path_msg.poses.append(pose)

        publisher.publish(path_msg)

    def publish_cmd_vel(self, norm_v_t, angle_v_t):
        cmd_vel = Twist()

        zeta = self.convert_angle(0) - self.convert_angle(angle_v_t)
        v_t_control = norm_v_t*np.cos(zeta)
        omega_control = -zeta/(6*5*0.01)

        cmd_vel.linear.x = norm_v_t
        cmd_vel.angular.z = angle_v_t

        self.cmd_vel_publisher.publish(cmd_vel)

        # what is this ???????
        with open("planner_log_With.txt", 'a') as f:
            f.write(f"{self.global_goal_x} {self.global_goal_y} {self.local_goal_x} {self.local_goal_y} {self.current_x} {self.current_y} {norm_v_t} {angle_v_t} \n")
        rospy.loginfo(f"Published to cmd_vel {cmd_vel.linear.x} {cmd_vel.angular.z}")
        return
    
    def publish_zero_cmd_vel(self):
        cmd_vel = Twist()

        cmd_vel.linear.x = 0
        cmd_vel.linear.z = 0

        self.cmd_vel_publisher.publish(cmd_vel)
        return
    
    def convert_angle(self, angle):
        angle = np.unwrap(np.array([angle]), discont=np.pi, axis=0, period= 6.283185307179586)
        return angle
    
    def update_local_goal(self):
        try:
            transform = self.tf_buffer.lookup_transform("base_link","map", rospy.Time(0), rospy.Duration(1.0))
            goal_in_base_link = do_transform_pose(self.global_goal_msg, transform)
            self.global_goal_x = self.global_goal_msg.pose.position.x
            self.global_goal_y = self.global_goal_msg.pose.position.y
            self.local_goal_x = goal_in_base_link.pose.position.x
            self.local_goal_y = goal_in_base_link.pose.position.y

        except tf2_ros.TransformException as e:
            rospy.logfatal(f"Could not transform goal in map frame to goal in local frame: {e}")
            return
        
    def goal_callback(self, global_goal_msg):
        rospy.loginfo("Recieved goal")
        self.goal_reached = False
        self.global_goal_msg = global_goal_msg
        self.update_local_goal()
        self.rollout_num = 0

    def plan(self):
        if not self.goal_reached:
            if not hasattr(self, 'current_x'):
                rospy.logerr("Not recieved current position x")
                self.publish_zero_cmd_vel()
                return
            if not hasattr(self, 'current_y'):
                rospy.logerr("Not recievded current position y")
                self.publish_zero_cmd_vel()
                return
            if not hasattr(self, 'global_goal_x'):
                rospy.logerr("Not recieved goal position x")
                return
            if not hasattr(self, 'global_goal_y'):
                rospy.logerr("Note recieved goal psoition y")
                return
            if not hasattr(self, 'local_goal_x'):
                rospy.logerr("Not recieved local goal x")
                return
            if not hasattr(self, 'local_goal_y'):
                rospy.logerr("Not recieved local goal y")
                return
            
        current_x = self.current_x
        current_y = self.current_y
        self.update_local_goal()

        state_initial = State(0, 0.1, self.current_vx, self.current_vy, 0, 0)
        state_goal = State(self.local_goal_x, self.local_goal_y)
        rospy.loginfo(f"Global Position {current_x} {current_y}")
        rospy.loginfo(f"Global Goal {self.global_goal_x} {self.global_goal_y}")
        rospy.loginfo(f"Local Goal {self.local_goal_x} {self.local_goal_y}")
        c_x, c_y, norm_v_t, angle_v_t = self.infer_trajectories(state_initial, state_goal)
        self.publish_cmd_vel(norm_v_t,angle_v_t)

        self.rollout_num += 1
        if (self.global_goal_x-self.current_x)**2 + (self.global_goal_y-self.current_y)**2 < 0.5:
            self.goal_reached=True
        else:
            self.publish_zero_cmd_vel()

def main(args=None):
    ClosedLoopSimulation()

if __name__ == '__main__':
    main()            
