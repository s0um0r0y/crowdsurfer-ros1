import torch
from torchinfo import summary
from torch import nn

# This processes occupancy grids (e.g., 60x60 grayscale maps showing free/occupied cells).
class OccupancyGridEmbedding(nn.Module):
    def __init__(self, embedding_dim=32):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Convolutional layers with ReLU activation and batch normalization.
        # Input: 1-channel image.
        # Output: 32 channels.
        # Kernel size: 3x3
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(32))
        
        # Deeper layers increase channels while keeping kernel size = 3
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(64))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(128))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 3),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(256))
        
        # AdaptiveAvgPool2d(1): Pool to 1x1 spatial resolution (global average).
        # Flatten: Turns (B, 256, 1, 1) → (B, 256).
        # Two fully connected layers to project into the final embedding_dim.
        self.output_layer = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                          nn.Flatten(),
                                          nn.Linear(256, 128),
                                          nn.ReLU(),
                                          nn.Dropout(), 
                                          nn.Linear(128, embedding_dim))
        
    def forward(self, x):

        # Simple pass through conv layers + output head
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.output_layer(out)

        return out

class DynamicObstacleEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim=32):

        # Processes time-series of dynamic obstacles, shaped like (B, N, 4, 10) where:
        # B = batch,
        # N = number of obstacles,
        # 4 = obstacle features (e.g., x, y, vx, vy),
        # 10 = time steps.
        super().__init__()

        self.embedding_dim = embedding_dim
        
        self.conv1 = nn.Sequential(nn.Conv1d(4, 32, 1),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(32))
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, 1),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(64))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, 1),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(128))
        self.conv4 = nn.Sequential(nn.Conv1d(128, 256, 1),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(256))
        self.output_layer = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                          nn.Flatten(),
                                          nn.Linear(256, 128),
                                          nn.ReLU(),
                                          nn.Dropout(), 
                                          nn.Linear(128, embedding_dim))

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.output_layer(out)

        return out

class ObservationEmbedding(torch.nn.Module):

    # Combines:
    # Occupancy grid embedding,
    # Dynamic obstacle embedding,
    # Heading (1D orientation value).
    def __init__(self, embedding_dim=16):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.occupancy_map_embedding = OccupancyGridEmbedding(self.embedding_dim)
        self.dynamic_obstacle_embedding = DynamicObstacleEmbedding(self.embedding_dim)
        self.heading_embedding = nn.Sequential(nn.Linear(1, embedding_dim//8),
                                               nn.ReLU())

        # Processes sequence of obstacle embeddings.
        # batch_first=True: input shape is (B, N, embedding_dim).
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=256,
                            num_layers=1,
                            batch_first=True, 
                            bidirectional=False)
        
        # Compress LSTM hidden state to embedding_dim.
        self.lstm_output_layer = nn.Linear(256, embedding_dim)

        # Combine all 3 embeddings (map, obstacles, heading) → final vector.
        self.combination_layer = nn.Sequential(nn.Linear(embedding_dim*2+embedding_dim//8, embedding_dim),
                                               nn.ReLU()) 

    def forward(self, occuapancy_map, dynamic_obstacles, heading):
        # occupancy_map: (B, 1, H, W)
        # dynamic_obstacles: (B, N, 4, 10)
        # heading: (B, 1)

        occupancy_map_embedding = self.occupancy_map_embedding(occuapancy_map)

        # Reshape obstacles to pass through CNN.
        # Output: (B, N, embedding_dim)
        B, N, _, _ = dynamic_obstacles.shape
        dynamic_obstacles_embedding = self.dynamic_obstacle_embedding(dynamic_obstacles.view(-1, 4, 10)).view(B, N, -1)
        dynamic_obstacles_embedding, _ = self.lstm(dynamic_obstacles_embedding)

        # Take output at first time step
        dynamic_obstacles_embedding = self.lstm_output_layer(dynamic_obstacles_embedding)[:, 0, :]

        heading_embedding = self.heading_embedding(heading)
        
        # Combines everything into one vector of shape (B, embedding_dim)
        observation_embedding = self.combination_layer(torch.concat((occupancy_map_embedding, dynamic_obstacles_embedding, heading_embedding), dim=1))        

        return observation_embedding  #, occupancy_map_embedding, dynamic_obstacles_embedding, heading_embedding

if __name__ == "__main__":
    model = ObservationEmbedding(embedding_dim=16)
    summary(model, [(4, 1, 60, 60), (4, 5, 4, 10), (4, 1)])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dummy data for testing
    o, a, b, c = model(torch.randn((4, 1, 60, 60)).to(device), torch.randn((4, 5, 4, 10)).to(device), torch.randn((4, 1)).to(device))
    print(f'Observation Embedding: {o.shape}, Occuapncy Grid: {a.shape}, Dynamic Obstacle: {b.shape}, Heading: {c.shape}')