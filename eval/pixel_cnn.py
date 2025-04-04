import torch
from tqdm import tqdm
from termcolor import colored
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from dataloaders.pixel_cnn import PixelCNNDataset

from models.vq_vae import VQVAE
from models.fused import FusedModel

from utils.save import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_root = 'data/processed_sim_bags'

dataset = PixelCNNDataset(root_dir='data/processed_sim_bags')
print("Total Dataset Size:", len(dataset))

# Subset Selection: Only takes the last 20% of the dataset (typically for validation/testing).
dataset = Subset(dataset, range(int(0.8*len(dataset)), len(dataset)))
print("Taking Traing Subset:", len(dataset))
print(colored(f"Loaded PixelCNN Dataset with {len(dataset)} samples", 'green'))
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

# Initializes the VQ-VAE model and moves it to GPU/CPU.
# Loads pre-trained weights from 'checkpoints/state_dict/vqvae.pth'.
# Calls eval() to set the model to inference mode (disables dropout, batch norm updates) to disable training-specific layers.

vqvae = VQVAE()
vqvae = vqvae.to(device)
vqvae.load_state_dict(torch.load('checkpoints/state_dict/vqvae.pth'))
vqvae.eval()

pixelcnn = FusedModel()
pixelcnn = pixelcnn.to(device)
pixelcnn.load_state_dict(torch.load('checkpoints/state_dict/pixelcnn.pth'))
pixelcnn.eval()

# Iterates through the dataset with a progress bar (tqdm).
for i, batch in enumerate(tqdm(dataloader)):
    #data io
    occupancy_grid, dynamic_obstacles, heading, coeffecients = batch

    # Extracts only the first timestep of the occupancy grid.
    # Adds a channel dimension (unsqueeze(1)) to match the expected (batch_size, channels, height, width) format.    
    occupancy_grid = occupancy_grid[:, 0, :, :].unsqueeze(1).float().to(device)
    
    dynamic_obstacles = dynamic_obstacles.permute(0, 1, 3, 2)[:, :, 1:, :].float().to(device)

    # Extracts the first timestep of heading.
    heading = heading[:, 0].unsqueeze(-1).float().to(device)

    # Swaps dimensions (permute(0, 2, 1)) to match model input.
    coeffecients = coeffecients[:, 0, : ,:].permute(0, 2, 1).to(device)

    # Replace NaN values with 0.
    occupancy_grid[torch.isnan(occupancy_grid)] = 0
    dynamic_obstacles[torch.isnan(dynamic_obstacles)] = 0
    heading[torch.isnan(heading)] = 0

    #forward
    # Passes input through PixelCNN to get an embedding (pixelcnn_embedding).
    pixelcnn_embedding = pixelcnn(occupancy_grid, dynamic_obstacles, heading).permute(0, 2, 1)

    # extracts the index of the most probable token (discrete representation).
    # pixelcnn_idx is the quantized representation of the trajectory.
    _, pixelcnn_idx = torch.max(pixelcnn_embedding, dim=1)
    # print("PixelCNN idx:", pixelcnn_idx)

    # Decodes the PixelCNN output using VQ-VAE to generate a predicted trajectory.
    pred_traj = vqvae.from_indices(pixelcnn_idx)

    # Uses VQ-VAE to encode ground truth trajectories into discrete embeddings
    with torch.no_grad():
        # Converts GT(Ground truth) trajectory into latent space.
        vqvae_embedding = vqvae.get_embedding(coeffecients)

        # Retrieves quantized indices (discrete representation).
        vqvae_idx = vqvae.get_indices(vqvae_embedding)
        # print("VQVAE idx:", vqvae_idx)
    
    gt_traj = coeffecients
    save_pixelcnn(occupancy_grid, dynamic_obstacles, gt_traj, pred_traj, filename=f'outputs/pixelcnn/{i}.png')
    # exit()