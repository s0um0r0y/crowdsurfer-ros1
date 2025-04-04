import os
import torch
import numpy as np
from tqdm import tqdm
from termcolor import colored
from torch.utils.data import Dataset, DataLoader

coefficients_file = 'best_priest_coefficients.npy' #train on elite priest
expert_trajectory_file = 'expert_trajectory.npy'

class VQVAEDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.index_file = os.path.join(root_dir, 'index.txt')

        with open(self.index_file, 'r') as f:
            self.samples = [line.split() for line in f.readlines()]

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        coefficients = torch.tensor(np.load(os.path.join(self.root_dir, str(self.samples[index][0]), coefficients_file)), dtype=torch.float32)

        # Reshapes coefficients to have a 3D shape (batch_size, num_coefficients, num_dimensions).
        coefficients = coefficients.view(-1, coefficients.shape[-2], coefficients.shape[-1])  
        expert_trajectory = torch.tensor(np.load(os.path.join(self.root_dir, str(self.samples[index][0]), expert_trajectory_file)))
        return coefficients, expert_trajectory

if __name__ == '__main__':
    dataset = VQVAEDataset(root_dir='data/processed_sim_bags')
    print(f'Loaded VQ-VAE Dataset with {len(dataset)} samples')

    # Wraps the dataset in a DataLoader, which allows batch processing.
    # Uses batch_size=1, meaning one sample per batch.
    # shuffle=False ensures samples are loaded sequentially.
    # num_workers=0 means data loading happens in the main process.

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(colored("Starting Dry Run on VQ-VAE Dataset", 'yellow'))
    for batch in tqdm(dataloader):
        continue
    print(colored('Dry run successful', 'green'))