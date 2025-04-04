import torch
from torchinfo import summary
from termcolor import colored

# The Encoder is a 1D convolutional neural network (CNN) that extracts features from an input signal.

'''
Architecture
Input: A tensor of shape [batch,input_channels,sequence_length].

Layers:

Input Layer: A convolutional layer followed by batch normalization and ReLU activation.
Hidden Layers: A sequence of convolutional layers with batch normalization and ReLU activation.
Output Layer: A final convolutional layer with batch normalization (no activation).
Output: Feature embeddings from the last convolutional layer.

Key Points
Uses 1D convolutions, meaning it works on sequential data.
The number of hidden layers is configurable.
The output preserves the number of channels but transforms the input sequence into a more structured latent representation.

The Encoder takes an input signal (e.g., a time-series or an image), processes it using convolutional layers, and outputs a feature representation.
'''
class Encoder(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 output_channels,
                 kernel_size,
                 num_hidden_layers,
                 embedding_size, 
                 embedding_dim):
        
        # Calls the parent class constructor and stores the number of hidden layers.
        super().__init__()

        self.num_hidden_layers = num_hidden_layers

        self.input_layer = torch.nn.Sequential(
            # A 1D convolutional layer processes the input.
            torch.nn.Conv1d(input_channels, hidden_channels, kernel_size, padding=5),

            # Batch normalization normalizes the output of the convolutional layer.
            # Batch normalization stabilizes training.
            torch.nn.BatchNorm1d(hidden_channels),

            # ReLU activation function introduces non-linearity.
            torch.nn.ReLU()
        )

        # torch.nn.ModuleList() is used to store multiple layers.
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=5),
                                                   torch.nn.BatchNorm1d(hidden_channels),
                                                   torch.nn.ReLU()) 
                                                   for __ in range(num_hidden_layers)])
        
        # The last convolutional layer processes the output before passing it to the vector quantization step.
        self.output_layer = torch.nn.Sequential(
            torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=5),
            torch.nn.BatchNorm1d(hidden_channels),
        )

    # Returns the encoded feature representation.
    def forward(self, x):
        out = self.input_layer(x)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.output_layer(out)
        return out

class VectorQuantizer(torch.nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_embeddings,
                 beta):
        super().__init__()

        # The dimensionality of each code vector.
        self.embedding_dim = embedding_dim

        # The number of code vectors.Number of discrete codebook vectors.
        self.num_embeddings = num_embeddings

        # A weighting factor for commitment loss
        self.beta = beta

        # Defines the codebook.
        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)

    def loss(self, x, quantized):

        # Ensures encoder outputs stay close to the quantized values
        commitment_loss = torch.nn.functional.mse_loss(quantized.detach(),x)

        # Ensures the codebook values are close to the encoder outputs
        codebook_loss = torch.nn.functional.mse_loss(x.detach(), quantized)

        # Final loss formulation
        loss = codebook_loss + self.beta*commitment_loss
        return loss

    def forward(self, x):

        # torch.cdist(x, self.embedding.weight): Computes the distance between x and all embeddings.
        # torch.argmin(): Returns the index of the closest embedding.
        indices = torch.argmin(torch.cdist(x.squeeze(-1), self.embedding.weight), dim=-1)

        # Retrieves the corresponding embedding.
        quantized = self.embedding.forward(indices)
        return quantized, indices
    
    def from_indices(self, indices):

        # Retrieves embeddings based on given indices.
        quantized = self.embedding.forward(indices)
        return quantized

# The Decoder reconstructs the original input from the quantized embedding.  
class Decoder(torch.nn.Module):
    def __init__(self, 
                 input_features,
                 hidden_features,
                 output_features,
                 num_hidden_layers):
        super().__init__()

        # Converts the flattened embedding back into meaningful features.
        self.input_layer = torch.nn.Sequential(torch.nn.Linear(input_features, hidden_features), 
                                                torch.nn.BatchNorm1d(hidden_features),
                                                torch.nn.ReLU())

        # Applies fully connected layers with ReLU activation.
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(hidden_features, hidden_features), 
                                                    torch.nn.BatchNorm1d(hidden_features),
                                                    torch.nn.ReLU()) 
                                                    for __ in range(num_hidden_layers)])

        # Outputs the final reconstruction.
        self.output_layer = torch.nn.Linear(hidden_features, output_features)

    def forward(self, x):
        out = self.input_layer(x)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.output_layer(out)
        return out

# The VQVAE model combines the Encoder, VectorQuantizer, and Decoder to form a complete architecture.    
class VQVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Sets model hyperparameters.
        self.kernel_size = 11
        self.hidden_channels = 96
        self.embedding_dim = 11
        self.embedding_size = 16
        self.output_channels = 96

        self.encoder = Encoder(input_channels=2,
                               hidden_channels=96,
                               output_channels=96,
                               kernel_size=11,
                               num_hidden_layers=8,
                               embedding_size=16,
                               embedding_dim=11)
        
        self.quantizer = VectorQuantizer(embedding_dim=self.embedding_dim,
                                         num_embeddings = self.embedding_size,
                                         beta=0.2)
        
        self.decoder = Decoder(input_features=96*11, 
                               hidden_features=1024,
                               output_features=22,
                               num_hidden_layers=8)

    def forward(self, x):

        # Encodes x into a latent representation
        embedding = self.encoder(x).squeeze(-1)

        # Quantizes it using the vector quantizer.
        quantized_embedding, __ = self.quantizer(embedding)

        # Applies a straight-through estimator (.detach() trick).
        quantized_embedding = embedding + (quantized_embedding - embedding).detach()
        quantizer_loss = self.quantizer.loss(embedding, quantized_embedding)

        reconstructed_output = self.decoder(quantized_embedding.reshape(quantized_embedding.shape[0], -1))

        return reconstructed_output, quantizer_loss
    
    def get_indices(self, embedding):
        _ , indices = self.quantizer(embedding)

        return indices

    def get_embedding(self, x):
        return self.encoder(x).squeeze(-1)

    def from_indices(self, indices):
        quantized_embedding = self.quantizer.from_indices(indices)
        out = self.decoder(quantized_embedding.reshape(quantized_embedding.shape[0], -1))
        return out

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    input_shape = [10, 2, 11]
    model = VQVAE().to(device)
    print(colored("Intialized VQ-VAE Model", 'green'))
    summary(model, input_shape)