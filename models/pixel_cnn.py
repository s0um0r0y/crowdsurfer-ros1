import torch
from torchinfo import summary
from torch import nn

from models.observation_embedding import ObservationEmbedding

# MASK_TYPES = ["A", "B"]: Type A is used for the first layer, while B is used for all others.
class MaskedConv1D(torch.nn.Module):
    MASK_TYPES = ["A", "B"]

    def __init__(self, mask_type: str, conv1d: nn.Conv1d):
        super().__init__()
        assert mask_type in self.MASK_TYPES

        self.conv1d = conv1d

        # register_buffer() ensures that mask is part of the model but not treated as a learnable parameter
        self.register_buffer("mask", self.conv1d.weight.data.clone())

        # Gets the kernel size (size) from the weight dimensions
        self.mask: torch.Tensor
        __, __, size = self.conv1d.weight.size()

        # Fills the mask with 1s
        self.mask.fill_(1)

        # Zeroes out the right half of the kernel to enforce causality
        # For type A: Blocks the entire center and right side.
        # For type B: Blocks only the right side.
        self.mask[:, :, size//2 + (mask_type == 'B'):] = 0

    def forward(self, x):

        # Applies the mask before using the convolution layer.
        self.conv1d.weight.data *= self.mask

        # Performs convolution on x
        output = self.conv1d.forward(x)
        return output

# This layer extends MaskedConv1D by adding gating mechanisms and conditional embeddings
class ConditionedGatedMaskedConv1D(torch.nn.Module):
    def __init__(self, 
                 mask_type,
                 input_channel,
                 output_channel,
                 kernel,
                 padding,
                 conditional_embedding_dim,
                 bias=False):
        super().__init__()

        # Defines the first masked convolution (main signal).
        self.masked_conv1d_1 = MaskedConv1D(mask_type,
                                            nn.Conv1d(in_channels=input_channel,
                                                      out_channels=output_channel,
                                                      kernel_size=kernel,
                                                      padding=padding,
                                                      bias=bias))
        
        # Defines the second masked convolution (gate signal).
        self.masked_conv1d_2 = MaskedConv1D(mask_type,
                                            nn.Conv1d(in_channels=input_channel,
                                                      out_channels=output_channel,
                                                      kernel_size=kernel,
                                                      padding=padding,
                                                      bias=bias))

        # Processes conditional embeddings for the first convolution.
        self.conditional_conv_1 = nn.Conv1d(in_channels=conditional_embedding_dim,
                                            out_channels=output_channel,
                                            kernel_size=kernel,
                                            padding=padding, 
                                            bias=bias)

        # Processes conditional embeddings for the gating mechanism.
        self.conditional_conv_2 = nn.Conv1d(in_channels=conditional_embedding_dim,
                                            out_channels=output_channel,
                                            kernel_size=kernel,
                                            padding=padding, 
                                            bias=bias)
    
    def forward(self, x, conditional_embedding):

        # tanh() to the main signal.
        input = nn.functional.tanh(self.masked_conv1d_1(x) + self.conditional_conv_1(conditional_embedding))

        # sigmoid() to the gate signal.
        gate = nn.functional.sigmoid(self.masked_conv1d_2(x) + self.conditional_conv_2(conditional_embedding))

        # Element-wise multiplication (input * gate) for controlled information flow.
        return input*gate

class PixelCNN(torch.nn.Module):
    def __init__(self,
                 num_embeddings=96,
                 conditional_embedding_dim=16,
                 input_channels=1,
                 kernel_size=3,
                 padding=1,
                 hidden_channels=256,
                 num_layers=16,
                 residual_connection_freq=4): # Adds residual connections every few layers.
        super().__init__()
        self.residual_connection_freq = residual_connection_freq

        # First masked convolution is type "A" (ensures independence).
        # Batch normalization improves stability
        self.input_layer = nn.ModuleList([ConditionedGatedMaskedConv1D(mask_type='A',
                                                                      input_channel=input_channels,
                                                                      output_channel=hidden_channels,
                                                                      kernel=kernel_size,
                                                                      padding=padding,
                                                                      conditional_embedding_dim=conditional_embedding_dim),
                                         nn.BatchNorm1d(hidden_channels)])
        
        self.layers = nn.ModuleList()

        # Stacks multiple masked convolutions (type "B").    
        for _ in range(num_layers):
            self.layers.extend([ConditionedGatedMaskedConv1D(mask_type='B',
                                                             input_channel=hidden_channels,
                                                             output_channel=hidden_channels,
                                                             kernel=kernel_size,
                                                             padding=padding,
                                                             conditional_embedding_dim=conditional_embedding_dim),
                                nn.BatchNorm1d(hidden_channels)])
        
        # Final 1D convolution layer maps hidden channels → output embeddings
        self.output_layer = nn.Conv1d(hidden_channels, num_embeddings, kernel_size=1)

    def forward(self, x, conditional_embedding):

        # Applies the input layer transformations
        for layer in self.input_layer:
            if isinstance (layer, ConditionedGatedMaskedConv1D):
                x = layer(x, conditional_embedding)
            else:
                x = layer(x)

        residual = x

        # Processes hidden layers.
        for i, layer in enumerate(self.layers):
            if isinstance (layer, ConditionedGatedMaskedConv1D):
                x = layer(x, conditional_embedding)
            else:
                x = layer(x)

            if (i+1)%self.residual_connection_freq == 0:
                x += residual
                residual = x
                
        x = self.output_layer(x)
        return x
    
if __name__=='__main__':
    model = PixelCNN()
    summary(model, ((4, 1, 16), (4, 16, 1)))