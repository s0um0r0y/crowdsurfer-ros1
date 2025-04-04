import torch
from tqdm import tqdm
from termcolor import colored
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from models.vq_vae import VQVAE
from models.fused import FusedModel

from dataloaders.pixel_cnn import PixelCNNDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_root = 'data/processed_sim_bags'

dataset = PixelCNNDataset(root_dir='data/processed_sim_bags')
print("Total Dataset Size:", len(dataset))

# Restricts to the first 80% of the dataset for training and logs size.
dataset = Subset(dataset, range(int(0.8*len(dataset))))
print("Taking Traing Subset:", len(dataset))
print(colored(f"Loaded PixelCNN Dataset with {len(dataset)} samples", 'green'))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

# Loads pretrained VQ-VAE model and sets it to evaluation mode (eval() disables dropout, etc.)
vqvae = torch.load('checkpoints/vqvae.pth').to(device)
vqvae.eval()

# Creates the PixelCNN model that uses observation embeddings for conditioning
pixelcnn = FusedModel().to(device)

# Loss function used to match PixelCNN outputs to VQ-VAE codebook indices
ce_loss_func = torch.nn.CrossEntropyLoss()

num_epochs = 500

# Optimizer: AdamW (Adam + weight decay)
optimizer = torch.optim.AdamW(pixelcnn.parameters(), lr=1e-3)

# Scheduler: Cosine Annealing (gradually reduces LR to 1e-6 over 500 epochs)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, 1e-6)

writer = SummaryWriter(log_dir='./pixelcnn_512')  # Initialize TensorBoard writer

for epoch in range(num_epochs):
    loss_list = []
    # tqdm shows a progress bar for the batches
    for batch in tqdm(dataloader):
        #data io

        # occupancy_grid: map image (B, 1, H, W)
        # dynamic_obstacles: sequence data (B, N, 4, 10)
        # heading: scalar (B, 1)
        # coeffecients: ground truth trajectory embeddings (B, 1, 16, T)
        occupancy_grid, dynamic_obstacles, heading, coeffecients = batch

        # Extracts the first channel and makes it shape (B, 1, H, W).
        occupancy_grid = occupancy_grid[:, 0, :, :].unsqueeze(1).float().to(device)

        # Rearranges dimensions from (B, N, 4, 10) → (B, N, 10, 4), then drops the first feature (index 0) → becomes (B, N, 10, 3)
        dynamic_obstacles = dynamic_obstacles.permute(0, 1, 3, 2)[:, :, 1:, :].float().to(device)

        # Extracts scalar heading and reshapes to (B, 1)
        heading = heading[:, 0].unsqueeze(-1).float().to(device)

        # (B, 1, 16, T) → drop middle dimension → (B, 16, T) → permute to (B, T, 16)
        coeffecients = coeffecients[:, 0, : ,:].permute(0, 2, 1).to(device)

        # Replaces NaN values (missing data) with zero
        occupancy_grid[torch.isnan(occupancy_grid)] = 0
        dynamic_obstacles[torch.isnan(dynamic_obstacles)] = 0
        heading[torch.isnan(heading)] = 0

        #forward
        # Feeds inputs to FusedModel, gets output logits.
        # Shape: (B, T, VocabSize) → permuted to match (B, T, Classes) for loss.
        pixelcnn_embedding = pixelcnn(occupancy_grid, dynamic_obstacles, heading).permute(0, 2, 1)
        # pixelcnn_logits = torch.nn.functional.softmax(pixelcnn_embedding, dim=2)

        with torch.no_grad():
            # Passes ground truth coefficients through VQ-VAE encoder:
            # get_embedding() → maps trajectory coefficients to continuous embedding space.
            # get_indices() → quantizes them to nearest codebook vector index.
            # vqvae_idx: shape = (B, T) with class IDs (0 to N)
            vqvae_embedding = vqvae.get_embedding(coeffecients)
            vqvae_idx = vqvae.get_indices(vqvae_embedding)
            # vqvae_one_hot = torch.nn.functional.one_hot(vqvae_idx, num_classes=16).float()

        # cross entropy loss function:
        loss = ce_loss_func(pixelcnn_embedding, vqvae_idx)

        # Backpropagates and updates model weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # records loss
        loss_list.append(loss.detach().cpu().item())
    
    avg_loss = sum(loss_list) / len(loss_list)
    print(f"Epoch {epoch} Loss: {avg_loss}")
    writer.add_scalar("Loss/train", avg_loss, epoch)     

torch.save(pixelcnn, f='checkpoints/pixelcnn_512.pth')
torch.save(pixelcnn.state_dict(), f='checkpoints/pixelcnn_512.pth')
