import os
import torch 
import numpy as np
from tqdm import tqdm
from termcolor import colored
from torch.utils.data import Dataset, DataLoader

coefficients_file = 'best_priest_coefficients.npy'
expert_trajectory_file = 'expert_trajectory.npy'
occupancy_grid_file = 'occupancy_map.npy'
dynamic_obstacles_file = 'dynamic_obstacle_metadata.npy'
heading_to_goal_file = 'goal_position.npy'

class PixelCNNDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        # index.txt contents are : bag_1  bag_scenario_name  num_timesteps
        self.index_file = os.path.join(root_dir, 'index.txt')

        with open(self.index_file, 'r') as f:
            self.bags = [line.split() for line in f.readlines()]

        # The dataset expands from bags to individual timesteps: If bag_1 has 10 timesteps, it contributes 10 samples.
        self.samples = []
        for bag in self.bags:
            for step in range(int(bag[2])):

                # Append the bag name and timestep to the samples list.
                self.samples.append([bag[0], step])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index): 
        occupancy_grid_data = []
        dynamic_obstacle_data = []
        coefficients_data = []
        heading_data = []

        # Extracts the current timestep (step) from the sample.
        step = self.samples[index][1]

        for i in range(min(step+1, 5)):   

            '''
            Loads up to 5 past timesteps (including the current timestep).
            Uses step+1, 5 to ensure it doesn't go beyond available steps.
            Occupancy grid: Extracts and appends a 2D grid of obstacles.
            Dynamic obstacles: Extracts and appends information about moving obstacles.
            Heading to goal: Extracts (x, y) direction and converts it to an angle using atan2(y, x).
            Coefficients: Extracts and appends PRIEST trajectory coefficients.   
            '''

            #get occupancy grid data
            occupancy_grid_step = np.load(os.path.join(self.root_dir, self.samples[index-i][0], occupancy_grid_file))[int(self.samples[index-i][1])]
            occupancy_grid_data.append(torch.tensor(occupancy_grid_step))

            #get dynamic obstacle data
            dynamic_obstacle_step = np.load(os.path.join(self.root_dir, self.samples[index-i][0], dynamic_obstacles_file))[int(self.samples[index-i][1])]
            dynamic_obstacle_data.append(torch.tensor(dynamic_obstacle_step))

            #get heading to goal
            heading_step = np.load(os.path.join(self.root_dir, self.samples[index-i][0], heading_to_goal_file))[int(self.samples[index-i][1])]
            heading = torch.tensor(heading_step)
            heading_data.append(torch.atan2(heading[1], heading[0]))
            
            #get coefficients data 
            coefficients_step = np.load(os.path.join(self.root_dir, self.samples[index-i][0], coefficients_file))[int(self.samples[index-i][1])]
            coefficients_data.append(torch.tensor(coefficients_step))

        # Converts lists into tensors.
        # Shapes before stacking: [(H, W), (H, W), (H, W)]
        # After stacking: (num_timesteps, H, W)
        occupancy_grid_data = torch.stack(occupancy_grid_data, dim=0)
        dynamic_obstacle_data = torch.stack(dynamic_obstacle_data, dim=0)
        heading_data = torch.stack(heading_data, dim=0)
        coefficients_data = torch.stack(coefficients_data, dim=0)

        # Padding for missing timesteps.
        missing_dims = 5-occupancy_grid_data.shape[0]

        # If there are fewer than 5 timesteps, it calls pad_empty_timesteps() to add zero-filled padding.
        if missing_dims>0:
            occupancy_grid_data = self.pad_empty_timesteps(occupancy_grid_data)
            dynamic_obstacle_data = self.pad_empty_timesteps(dynamic_obstacle_data)
            heading_data = self.pad_empty_timesteps(heading_data)
            coefficients_data = self.pad_empty_timesteps(coefficients_data)

        return occupancy_grid_data, dynamic_obstacle_data, heading_data, coefficients_data
    
    # Creates a tensor of zeros with missing timesteps and appends it.
    # Ensures all samples have exactly 5 timesteps.
    def pad_empty_timesteps(self, sample, num_timesteps=5):
        channels_to_pad = num_timesteps - sample.shape[0]  # 5 - 3 = 2
        pad_size = []
        pad_size.append(channels_to_pad)
        pad_size.extend(sample.shape[1:])
        pad_sample = torch.zeros(pad_size, dtype=sample.dtype, device=sample.device)
        return torch.cat([sample, pad_sample], dim=0)
    
if __name__ == "__main__":
    data = PixelCNNDataset(root_dir="./data/processed_sim_bags/")
    dataloader = DataLoader(data, batch_size=1, shuffle=True)

    # Iterates over the dataset to verify that samples are loaded correctly.
    for i in tqdm(dataloader):
        print(f"Occupancy Grid: {i[0].shape}, Dynamic Obstacles: {i[1].unsqueeze(2).shape}, Heading Data: {i[2].shape}, Coefficients Data: {i[3].shape}")