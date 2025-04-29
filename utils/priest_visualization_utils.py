import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_plan(start, goal, obstacles, trajectory_x, trajectory_y, filename):

    static_x = obstacles.obstacle_trajectory_x[:,0]
    static_y = obstacles.obstacle_trajectory_y[:,0]

    dynamic_x = obstacles.dynamic_obstacle_trajectory_x[:,0]
    dynamic_y = obstacles.dynamic_obstacle_trajetory_y[:,0]

    fig, ax = plt.subplots(figsize=(8,8))

    ellipse_start = Rectangle((start.x, start.y), width=0.15, height=0.15, edgecolor='green', facecolor='green')
    ellipse_final = Rectangle((goal.x, goal.y), width=0.15, height=0.15, edgecolor='red', facecolor='red')

    ax.add_patch(ellipse_start)
    ax.add_patch(ellipse_final)

    # Plot static obstacles as ellipses (size 0.05m by 0.005m)
    for x,y in zip(dynamic_x, dynamic_y):
        ellipse = Rectangle((x, y), width=0.68, height=0.68, edgecolor='orange',facecolor='none', lw=2)
        ax.add_patch(ellipse)

    # plot the trajectories
    ax.set_xlim(0,10)
    ax.set_ylim(0,10)
    ax.set_aspect('equal')
    ax.set_title("Static and Dynamic Obstacles")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.grid(True)

    # Save the plot as an image
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)