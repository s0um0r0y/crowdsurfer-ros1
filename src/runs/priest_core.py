from os import wait
import jax
import jax.numpy as jnp
import jax.random as random

import numpy as np
import torch
from bernstein_coeff_order10 import generate_order_10_bernstein_coefficients

import os
import shutil

from utils.priest_visualization_utils import *

class State:
    def __init__(self, 
                 x, 
                 y, 
                 vx = None, 
                 vy = None, 
                 ax = None, 
                 ay = None, 
                 normal_x = None, 
                 normal_y = None):
        
        # Position
        self.x = x
        self.y = y

        # Velocity
        self.vx = vx
        self.vy = vy

        # Acceleration
        self.ax = ax
        self.ay = ay

        # Normal vector
        self.normal_x = normal_x
        self.normal_y = normal_y

class Obstacles:
    def __init__(self, 
                 static_obstacles_x,
                 static_obstacles_y,
                 dynamic_obstacles_x,
                 dynamic_obstacles_y,
                 dynamic_obstacles_vx,
                 dynamic_obstacles_vy,
                 time_horizon=10,
                 num_steps=1000):
        
        self.num_obstacles = 40
        self.num_obstacles_projected = 10

        self.static_obstacles_x = torch.tensor(static_obstacles_x)
        self.static_obstacles_y = torch.tensor(static_obstacles_y)

        self.dynamic_obstacles_x = torch.tensor(dynamic_obstacles_x)
        self.dynamic_obstacles_y = torch.tensor(dynamic_obstacles_y)

        self.dynamic_obstacles_vx = torch.tensor(dynamic_obstacles_vx)
        self.dynamic_obstacles_vy = torch.tensor(dynamic_obstacles_vy)

        self.timesteps = jnp.linspace(0, time_horizon, num_steps)
        self.obstacle_trajectory_x = static_obstacles_x[:, None] * jnp.ones((1, num_steps))
        self.obstacle_trajectory_y = static_obstacles_y[:, None] * jnp.ones((1, num_steps))

        distances = jnp.sqrt(static_obstacles_x**2 + static_obstacles_y**2)
        sorted_indices = jnp.argsort(distances).flatten()

        self.obstacle_trajectory_x = self.obstacle_trajectory_x[sorted_indices[:self.num_obstacles],:]
        self.obstacle_trajectory_y = self.obstacle_trajectory_y[sorted_indices[:self.num_obstacles],:]

        self.obstacle_trajectory_projected_x = self.obstacle_trajectory_x[sorted_indices[:self.num_obstacles_projected],:]
        self.obstacle_trajectory_projected_y = self.obstacle_trajectory_y[sorted_indices[:self.num_obstacles_projected],:]

        self.dynamic_obstacles_trajectory_x = dynamic_obstacles_x[:, None] + dynamic_obstacles_vx[:, None] * self.timesteps
        self.dynamic_obstacles_trajectory_y = dynamic_obstacles_y[:, None] + dynamic_obstacles_vy[:, None] * self.timesteps

class Priest:
    def __init__(self):
        
        self.key = random.PRNGkey(0)
        self.time_horizon = 10
        self.velocity_desired = 1
        self.num_trajectory_steps = 1000

        self.v_max = 1
        self.a_max = 1

        self.rho_obstacle = 1
        self.rho_velocity = 1
        self.rho_acceleration = 1
        self.rho_projection = 1

        self.clearance_weight = 1
        self.obstacle_weight = 1.2
        self.smoothness_weight = 0.1
        self.track_weight = 0.2
        self.residual_norm_weight = 0.1

        self.alpha = 0.7
        self.lambda_ = 0.9

        self.cem_iterations = 11
        self.am_iterations = 1

        self.num_warm_trajectories_1 = 10
        self.num_warm_trajectories_2 = 10
        self.num_warm_trajectories = self.num_warm_trajectories_1 + self.num_warm_trajectories_2

        # obstacle configuration
        self.static_obstacle_A = 0.5
        self.static_obstacle_B = 0.5
        self.dynamic_obstacle_A = 0.68
        self.dynamic_obstacle_B = 0.68

        self.num_static_obstacles = 10
        self.num_dynamic_obstacles = 10

        # fixing this as the global generates the same perturbations every step
        self.scale_factor_warm = np.random.normal(0, 0.8, (self.num_warm_trajectories, 1))
        self.scale_factor_warm[0] = 0.0
        self.scale_factor_warm[-1] = 0.0

        self.P, self.Pdot, self.Pddot = generate_order_10_bernstein_coefficients(jnp.linspace(0, 
                                                                                              self.time_horizon, 
                                                                                              self.num_trajectory_steps), 
                                                                                              0, 
                                                                                              self.time_horizon)

        self.A_obstacle = jnp.tile(self.P, (45, 1))
        self.A_projected = jnp.dot(self.Pddot, self.Pddot.T)

        self.cost_matrix_inverse = jnp.linalg.inv(jnp.dot(self.P, self.P.T) + 0.0001 * jnp.identity(11) + 1*jnp.dot(self.Pddot, self.Pddot.T))

        A = jnp.vstack((self.P[:,0],
                        self.Pdot[:,0],
                        self.Pddot[:, 0],
                        self.P[:, -1])).T
        
        Q = jnp.linalg.inv(jnp.dot(jnp.identity(11), jnp.identity(11).T) + 0.0001 * jnp.dot(self.P, self.P.T) + jnp.dot(self.Pdot, self.Pdot.T) + jnp.dot(self.Pddot, self.Pddot.T))
        self.cost_matrix_inverse = jnp.linalg.inv(jnp.vstack((jnp.hstack((Q, A)),jnp.hstack((A.T , jnp.zeros((4, 4)))))))

        log_dir = "./logs/priest"

        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)

    def get_intermediate_position(self, 
                                  split, 
                                  x_waypoints, 
                                  y_waypoints, 
                                  x_diff, 
                                  y_diff, 
                                  cumulative_segment_lengths):
        
        # distance travelled in time horizon
        distance = self.velocity_desired * self.time_horizon * split
        
        point_index = jnp.argmin(jnp.abs(cumulative_segment_lengths - distance))
        normal_x = -y_diff[point_index]/jnp.sqrt(x_diff[point_index]**2 + y_diff[point_index]**2)
        normal_y = x_diff[point_index]/jnp.sqrt(x_diff[point_index]**2 + y_diff[point_index]**2)
        final_state = State(x=x_waypoints[point_index],
                            y=y_waypoints[point_index],
                            normal_x=normal_x,
                            normal_y=normal_y)
        return final_state

    def initialize_trajectories(self, state_initial, state_goal, obstacles):
        '''
        Initializes trajectories using the start and end positions for the optimization

        Inputs
        state_initial: state class that holds the data of the initial class
        state_final: State class that holds the data of the final class
        obstacles: Obstacle class that holds the static and dynamic obstacles data
        '''

        x_waypoints = jnp.linspace(state_initial.x, state_goal.x, self.num_trajectory_steps)
        y_waypoints = jnp.linspace(state_initial.y, state_goal.y, self.num_trajectory_steps)

        x_diff = jnp.diff(x_waypoints)
        y_diff = jnp.diff(y_waypoints)

        segment_lenghts = jnp.sqrt(x_diff**2 + y_diff**2)
        cumulative_segement_lengths = jnp.cumsum(segment_lenghts)

        state_25 = self.get_intermediate_position(0.25, x_waypoints, y_waypoints, x_diff, y_diff, cumulative_segement_lengths)
        state_50 = self.get_intermediate_position(0.5, x_waypoints, y_waypoints, x_diff, y_diff, cumulative_segement_lengths)
        state_75 = self.get_intermediate_position(0.75, x_waypoints, y_waypoints, x_diff, y_diff, cumulative_segement_lengths)
        state_final = self.get_intermediate_position(1, x_waypoints, y_waypoints, x_diff, y_diff, cumulative_segement_lengths)

        # compute a set of warm-start trajectories with start, mid and end points
        x_initial_matrice = state_initial.x * jnp.ones((self.num_warm_trajectories, 1))
        y_initial_matrice = state_initial.y * jnp.ones((self.num_warm_trajectories, 1))

        vx_initial_matrice = state_initial.vx * jnp.ones((self.num_warm_trajectories, 1))
        vy_initial_matrice = state_initial.vy * jnp.ones((self.num_warm_trajectories, 1))

        ax_initial_matrice = state_initial.ax * jnp.ones((self.num_warm_trajectories, 1))
        ay_initial_matrice = state_initial.ay * jnp.ones((self.num_warm_trajectories, 1))

        x_50_matrice = (state_50.x + self.scale_factor_warm)*jnp.ones((self.num_warm_trajectories,1))
        y_50_matrice = (state_50.y + self.scale_factor_warm)*jnp.ones((self.num_warm_trajectories,1))

        x_final_matrice = state_final.x * jnp.ones((self.num_warm_trajectories, 1))
        y_final_matrice = state_final.y * jnp.ones((self.num_warm_trajectories, 1))

        B_x = jnp.hstack((x_initial_matrice, vx_initial_matrice, ax_initial_matrice, x_50_matrice, x_final_matrice))
        B_y = jnp.hstack((y_initial_matrice, vy_initial_matrice, ay_initial_matrice, y_50_matrice, y_final_matrice))

        A = jnp.vstack((self.P[:, 0],
                        self.Pdot[:, 0],
                        self.Pddot[:, 0],
                        self.P[:, self.num_trajectory_steps//2],
                        self.P[:, -1])).T
        
        Q = jnp.dot(self.Pddot, self.Pddot.T)

        rhs_x = jnp.hstack((-jnp.zeros((self.num_warm_trajectories, 11)), B_x))
        rhs_y = jnp.hstack((-jnp.zeros((self.num_warm_trajectories, 11)), B_y))

        C = jnp.vstack((jnp.hstack((Q, A)), jnp.hstack((A.T, jnp.zeros((5, 5))))))
        C_inverse = jnp.stack([jnp.linalg.inv(C)] * self.num_warm_trajectories)

        c_x_warm1 = jax.vmap(jnp.matmul)(C_inverse, rhs_x)[:, :11]
        c_y_warm1 = jax.vmap(jnp.matmul)(C_inverse,  rhs_y)[:, :11]

        # compute a set of warm-start trajectories 
        x_25_matrice = (state_25.x + self.scale_factor_warm)*jnp.ones((self.num_warm_trajectories, 1))
        y_25_matrice = (state_25.y + self.scale_factor_warm)*jnp.ones((self.num_warm_trajectories, 1))

        x_75_matrice = (state_75.x + self.scale_factor_warm)*jnp.ones((self.num_warm_trajectories, 1))
        y_75_matrice = (state_75.y + self.scale_factor_warm)*jnp.ones((self.num_warm_trajectories, 1))
        
        B_x = jnp.hstack((x_initial_matrice, vx_initial_matrice, ax_initial_matrice, x_25_matrice, x_50_matrice, x_75_matrice, x_final_matrice))
        B_y = jnp.hstack((y_initial_matrice, vy_initial_matrice, ay_initial_matrice, y_25_matrice, y_50_matrice, y_75_matrice, x_final_matrice))

        A = jnp.vstack((self.P[:, 0],
                        self.Pdot[:, 0],
                        self.Pddort[:, 0],
                        self.P[:, self.num_trajectory_steps//4],
                        self.P[:, self.num_trajectory_steps//2],
                        self.P[:, 3*self.num_trajectory_steps//4],
                        self.P[:, -1])).T 
        
        rhs_x = jnp.hstack((-jnp.zeros((self.num_warm_trajectories, 11)), B_x))
        rhs_y = jnp.hstack((-jnp.zeros((self.num_warm_trajectories, 11)), B_y))

        C = jnp.vstack((jnp.hstack((Q, A)), jnp.hstack((A.T, jnp.zeros((7,7))))))
        C_inverse = jnp.stack([jnp.linalg.inv(C)*self.num_warm_trajectories])

        c_x_warm_2 = jax.vmap(jnp.matmul)(C_inverse, rhs_x)[:, :11]
        c_y_warm_2 = jax.vmap(jnp.matmul)(C_inverse, rhs_y)[:, :11]

        x_warm2 = jax.vmap(jnp.dot, in_axes=(0, None))(c_x_warm_2, self.P)
        y_warm2 = jax.vmap(jnp.dot, in_axes=(0, None))(c_y_warm_2, self.P)

        # compute obstacle cost
        static_obstacle_distance_x = x_warm2[:, None, :] - obstacles.obstacles_trajectory_x[None, :, :]
        static_obstacle_distance_y = y_warm2[:, None, :] - obstacles.obstacles_trajectory_y[None, :, :]

        dynamic_obstacle_distance_x = x_warm2[:, None, :] - obstacles.dynamic_obstacles_trajectory_x[None, :, :]
        dynamic_obstacle_distance_y = y_warm2[:, None, :] - obstacles.dynamic_obstacles_trajectory_y[None, :, :]

        cost_static_obstacles = -static_obstacle_distance_x**2/self.static_obstacle_A**2 - static_obstacle_distance_y**2/self.static_obstacle_B**2 + 1
        cost_dynamic_obstacles = -dynamic_obstacle_distance_x**2/self.dynamic_obstacle_A**2 - dynamic_obstacle_distance_y**2/self.dynamic_obstacle_B**2 + 1 

        cost_obstacles_stacked = jnp.hstack((cost_static_obstacles, cost_dynamic_obstacles))
        penalty_obstacles = jnp.linalg.norm(jnp.maximum(jnp.zeros_like(cost_obstacles_stacked), cost_obstacles_stacked), axis=(1, 2))
        idx_elite_trajectories = jnp.argsort(penalty_obstacles)

        c_x_warm_2 = c_x_warm_2[idx_elite_trajectories[:self.num_warm_trajectories_2], :]
        c_y_warm_2 = c_y_warm_2[idx_elite_trajectories[:self.num_warm_trajectories_2], :]

        # take some warm1 and some elite warm2 trajectories
        c_x_warm_2 = jnp.vstack((c_x_warm1, c_x_warm_2))
        c_y_warm_2 = jnp.vstack((c_y_warm1, c_y_warm_2))

        x_warm2 = jnp.matmul(c_x_warm_2, self.P)
        y_warm2 = jnp.matmul(c_y_warm_2, self.P)

        # calculate linear cost 
        cost_linear_x = jnp.matmul(x_warm2, self.P.T)
        cost_linear_y = jnp.matmul(y_warm2, self.P.T)

        c_x_warm = jnp.matmul(cost_linear_x, self.cost_matrix_inverse.T)
        c_y_warm = jnp.matmul(cost_linear_y, self.cost_matrix_inverse.T)

        # reconstruct full trajectory using the Bernstein polynomial
        x_warm = jnp.matmul(c_x_warm, self.P)
        y_warm = jnp.matmul(c_y_warm, self.P)

        xdot_warm = jnp.matmul(c_x_warm, self.Pdot)
        ydot_warm = jnp.matmul(c_y_warm, self.Pdot)

        xddot_warm = jnp.matmul(c_x_warm, self.Pddot)
        yddot_warm = jnp.matmul(c_y_warm, self.Pddot)

        # get the mean and covariance of the warm start trajectories
        c_mean = jnp.mean(jnp.hstack((c_x_warm, c_y_warm)), axis=0)
        c_cov = jnp.cov(jnp.hstack((c_x_warm, c_y_warm)).T)

        return c_mean, c_cov, c_x_warm, c_y_warm, x_warm, y_warm, xdot_warm, ydot_warm, xddot_warm, yddot_warm, state_final, x_waypoints, y_waypoints

    def project_trajectories(self, x, y, x_straight_line, y_straight_line):
        # this function computes the projected points of a given trajectory with respect to the straight line

        dist = jnp.sqrt((x_straight_line - x)**2 + (y_straight_line))
        idx = jnp.argmin(dist)

        x_project = x[idx]
        y_project = y[idx]
        return x_project, y_project
    
    def optimise_trajectories(self,
                              x_initial,
                              y_initial,
                              vx_initial,
                              vy_initial,
                              ax_initial,
                              ay_initial,
                              x_final,
                              y_final,
                              c_x, 
                              c_y,
                              x, 
                              y,
                              xdot, 
                              ydot, 
                              xddot, 
                              yddot,
                              static_obstacle_trajectory_x, 
                              static_obstacle_trajectory_y,
                              dynamic_obstacle_trajectory_x, 
                              dynamic_obstacle_trajectory_y):
        
        # runs AM to form the trajectory

        NUM_STATIC_OBSTACLES = static_obstacle_trajectory_x.shape[0]
        NUM_DYNAMIC_OBSTACLES = dynamic_obstacle_trajectory_x.shape[0]
        NUM_TOTAL_OBSTACLES = NUM_STATIC_OBSTACLES + NUM_DYNAMIC_OBSTACLES

        NUM_TRAJECTORIES = x.shape[0]

        A = jnp.vstack((self.static_obstacle_A*jnp.ones_like(static_obstacle_trajectory_x), 
                        self.dynamic_obstacle_A*jnp.ones_like(dynamic_obstacle_trajectory_x)))
        B = jnp.vstack((self.static_obstacle_B*jnp.ones_like(static_obstacle_trajectory_y), 
                        self.dynamic_obstacle_B*jnp.ones_like(dynamic_obstacle_trajectory_y)))

        obstacle_trajectory_x = jnp.vstack((static_obstacle_trajectory_x, dynamic_obstacle_trajectory_x))
        obstacle_trajectory_y = jnp.vstack((static_obstacle_trajectory_y, dynamic_obstacle_trajectory_y))

        delta_x = x[:, None, :] - obstacle_trajectory_x[None, :, :]
        delta_y = y[:, None, :] - obstacle_trajectory_y[None, :, :]

        alpha_obstacles = jnp.atan2(delta_y, delta_x)
        alpha_velocity = jnp.atan2(ydot, xdot)
        alpha_acceleration = jnp.atan2(yddot, xddot)

        d_obstacles = jnp.maximum(1,(A[None, :, :]*delta_x*jnp.cos(alpha_obstacles) + B[None, :, :]*delta_y*jnp.sin(alpha_obstacles))
                                  /((A[None, :, :]*jnp.cos(alpha_obstacles))**2 + (B[None, :, :]*jnp.sin(alpha_obstacles))**2))
        
        d_velocity = jnp.minimum(1, xdot*jnp.cos(alpha_velocity) + ydot*jnp.sin(alpha_velocity))
        d_acceleration = jnp.minimum(1, xddot*jnp.cos(alpha_acceleration) + yddot*jnp.sin(alpha_acceleration))

        lambda_x = jnp.zeros(())
        lambda_y = jnp.zeros(())

        b_boundary_x = jnp.stack([jnp.hstack(jnp.asarray((x_initial, vx_initial, ax_initial, x_final)))]*NUM_TRAJECTORIES)
        b_boundary_y = jnp.stack([jnp.hstack(jnp.asarray((y_initial, vy_initial, ay_initial, y_final)))]*NUM_TRAJECTORIES)

        for i in range(self.am_iterations):

            print("AM Iteration: ", i+1)
            print("Cx: ", c_x[0])

            b_x_projected = c_x
            b_y_projected = c_y

            b_vx = d_velocity*jnp.cos(alpha_velocity)
            b_vy = d_velocity*jnp.sin(alpha_velocity)

            b_ax = d_acceleration*jnp.cos(alpha_acceleration)
            b_ay = d_acceleration*jnp.sin(alpha_acceleration)

            b_x_obstacles = d_obstacles*jnp.cos(alpha_obstacles) + obstacle_trajectory_x
            b_x_obstacles = b_x_obstacles.reshape(-1, 1000)
            b_y_obstacles = d_obstacles*jnp.sin(alpha_obstacles) + obstacle_trajectory_y
            b_y_obstacles = b_y_obstacles.reshape(-1, 1000)

            linear_cost_x = jnp.dot(self.A_projected, b_x_projected.T) + jnp.dot(self.P, b_x_obstacles.T).reshape(-1, NUM_TRAJECTORIES, NUM_TOTAL_OBSTACLES).sum(-1) + jnp.dot(self.Pdot, b_vx.T) + jnp.dot(self.Pddot, b_ax.T) + lambda_x
            linear_cost_y = jnp.dot(self.A_projected, b_y_projected.T) + jnp.dot(self.P, b_y_obstacles.T).reshape(-1, NUM_TRAJECTORIES, NUM_TOTAL_OBSTACLES).sum(-1) + jnp.dot(self.Pdot, b_vy.T) + jnp.dot(self.Pddot, b_ay.T) + lambda_y

            b_x = jnp.concatenate((linear_cost_x.T, b_boundary_x), axis=1)
            b_y = jnp.concatenate((linear_cost_y.T, b_boundary_y), axis=1)

            c_x = jax.vmap(jnp.dot, in_axes=(None, 0))(self.cost_matrix_inverse, b_x)[:, :11]
            c_x = jax.vmap(jnp.dot, in_axes=(None, 0))(self.cost_matrix_inverse, b_x)[:, :11]

            if np.isnan(c_x).any():
                print("NaN in the c_x")
                exit()

            if np.isnan(c_y).any():
                print("NaN in the c_x")
                exit()

            x = jnp.matmul(c_x, self.P)
            y = jnp.matmul(c_y, self.P)

            xdot = jnp.matmul(c_x, self.Pdot)
            ydot = jnp.matmul(c_x, self.Pdot)

            xddot = jnp.matmul(c_x, self.Pddot)
            yddot = jnp.matmul(c_x, self.Pddot)

            alpha_obstacles = jnp.atan2(delta_y, delta_x) 
            alpha_velocity = jnp.atan2(ydot, xdot)
            alpha_acceleration = jnp.atan2(yddot, xddot)

            residuals_obstacles_x = ((delta_x) - d_obstacles*A*jnp.cos(alpha_obstacles)).reshape(-1, 1000)
            residuals_obstacles_y = ((delta_y) - d_obstacles*A*jnp.cos(alpha_obstacles)).reshape(-1, 1000)

            residuals_velocity_x = xdot - d_velocity*jnp.cos(alpha_velocity)
            residuals_velocity_y = xdot - d_velocity*jnp.cos(alpha_velocity)

            residuals_acceleration_x = xddot - d_acceleration*jnp.cos(alpha_acceleration)
            residuals_acceleration_y = yddot - d_acceleration*jnp.sin(alpha_acceleration)

            lambda_x = lambda_x - jnp.dot(self.P, residuals_obstacles_x.T).reshape(-1, NUM_TRAJECTORIES, NUM_TOTAL_OBSTACLES).sum(-1) - jnp.dot(self.Pdot, residuals_velocity_x.T) - jnp.dot(self.Pddot, residuals_acceleration_x.T)
            lambda_y = lambda_y - jnp.dot(self.P, residuals_obstacles_y.T).reshape(-1, NUM_TRAJECTORIES, NUM_TOTAL_OBSTACLES).sum(-1) - jnp.dot(self.Pdot, residuals_velocity_y.T) - jnp.dot(self.Pddot, residuals_acceleration_y.T)

            residual_norm_obstacle = jnp.linalg.norm(jnp.hstack((residuals_obstacles_x, residuals_obstacles_y)), axis=1).rehape(-1, 1000)
            residual_norm_velocity = jnp.linalg.norm(jnp.hstack((residuals_velocity_x, residuals_velocity_y)), axis=1)
            residual_norm_acceleration = jnp.linalg.norm(jnp.hstack((residuals_acceleration_x, residuals_acceleration_y)), axis=1)

            residual_norm = residual_norm_obstacle +  residual_norm_velocity

        return c_x, c_y, x, y, xdot, ydot, xddot, yddot, residual_norm
    
    def compute_cost(self,
                     x,
                     y,
                     xdot,
                     ydot,
                     xddot,
                     yddot,
                     x_project,
                     y_project,
                     residual_norm,
                     static_obstacles_trajectory_x, 
                     static_obstacles_trajectory_y,
                     dynamic_obstacles_trajectory_x,
                     dynamic_obstacles_trajectory_y):
        
        # calculates the cost of the trajectory

        dist_obs_static = - (x[:, None, :] - static_obstacles_trajectory_x[None, :, :])**2/self.static_obstacle_A**2 - (y[:, None, :] - static_obstacles_trajectory_y[None, :, :])**2/self.static_obstacle_B**2 + 1
        dist_obs_dynamic = - (x[:, None, :] - dynamic_obstacles_trajectory_x[None, :, :])**2/self.dynamic_obstacle_B**2 - (y[:, None, :] - dynamic_obstacles_trajectory_y[None, :, :])**2/self.dynamic_obstacle_B**2 + 1

        dist = jnp.hstack((dist_obs_dynamic, dist_obs_static))
        
        clearance = -jnp.min(dist, axis=1).sum(-1)
        obstacle = jnp.linalg.nomr(jnp.maximum(0, dist), axis = 1).sum(-1)
        smoothness = jnp.sqrt(xddot**2 + yddot**2).sum(-1)
        track = jnp.linalg.norm(x - x_project, axis=1)+ jnp.linalg.norm(y - y_project, axis=1)

        cost = self.clearance_weight*clearance + self.obstacle_weight*obstacle + self.smoothness_weight*smoothness + self.track_weight*track + self.residual_norm_weight*residual_norm

        return cost
    
    def update_distribution(self, c_elite_x, c_elite_y, c_mean, c_cov,cost):
        # updating the distribution of trajectories

        c_elite = jnp.concatenate((c_elite_x, c_elite_y), axis=1)

        beta = jnp.min(cost)
        d = jnp.exp(-cost * beta / self.lambda_)
        d = d / jnp.sum(d)

        c_mean_new = (1 - self.alpha)*c_mean + self.alpha*jnp.sum(c_elite * d[:, None], axis=0)
        
        # shape: (num_elites, 22)
        diff = c_elite - c_mean_new
        weighted_outer = jnp.einsum("ni,nj->ij", diff * d[:, None], diff)
        c_cov_new = (1 - self.alpha)*c_cov + self.alpha*(weighted_outer)

        return c_mean_new, c_cov_new
    
    def sample_trajectories(self, c_mean, c_cov):
        # sampling trajectories from the distribution

        N = 30
        c = jax.random.multivariate_normal(self.key, c_mean, c_cov, N)
        c_x = c[:, :11]
        c_y = c[:, :11]
        return c_x, c_y
    
    def compute_controls(self):
        # empty ?????

        return None
    
    def get_trajectory(self, start:State, goal:State, obstacles:Obstacles):
        c_mean, c_cov, c_x, c_y, x, y, xdot, ydot, xddot, yddot, state_final, x_straight_line, y_straight_line = self.initialize_trajectories(start, goal, obstacles)

        for i in range(self.cem_iterations):
            print("Running CEM iteration:", i+1)

            # run the AM optimization
            residual_norm = self.optimise_trajectories(start.x,
                                                       start.y,
                                                       start.vx,
                                                       start.vy,
                                                       start.ax,
                                                       start.ay,
                                                       state_final.x,
                                                       state_final.y,
                                                       c_x,
                                                       c_y,
                                                       x,
                                                       y,
                                                       xdot,
                                                       ydot,
                                                       xddot,
                                                       yddot,
                                                       obstacles.obstacle_trajectory_x,
                                                       obstacles.obstacle_trajectory_y)
            
            # project the trajectories
            x_project, y_project = self.project_trajectories(x, y, x_straight_line, y_straight_line)

            # compute the cost for the trajectories
            cost = self.compute_cost(x, y,
                                     xdot, ydot,
                                     xddot, yddot,
                                     x_project, y_project,
                                     residual_norm,
                                    obstacles.obstacle_trajectory_x, 
                                    obstacles.obstacle_trajectory_y,
                                     obstacles.dynamic_obstacles_trajectory_x,
                                     obstacles.dynamic_obstacles_trajectory_y)
            
            # take the best trajectories (ones with the lowest cost)
            elite_idx = jnp.argsort(cost, axis=0)
            c_elite_x = c_x[elite_idx[:10]]
            c_elite_y= c_y[elite_idx[:10]]

            # update the distribution using the best trajectories
            self.update_distribution(c_elite_x,c_elite_y,c_mean,c_cov,cost[elite_idx[:10]])

        return None
