"""
model.py
---------
Defines the Deep Q-Network (DQN) model using a Convolutional Neural Network (CNN)
for playing Atari Pong.
"""

import torch
import torch.nn as nn

class DQNCNN(nn.Module):
    """
    Deep Q-Network (CNN) used for estimating Q-values in Pong.

    Args:
        input_channels (int): Number of stacked frames (typically 4).
        n_actions (int): Number of valid Pong actions (typically 6).
    """
    def __init__(self, input_channels: int, n_actions: int):
        super().__init__()

        #  Convolutional layers to extract visual features from frames
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )

        # Helper to compute flattened conv output size dynamically
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(80, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        # Fully connected layers map extracted features to Q-values
        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN.

        Args:
            x (torch.Tensor): Batch of states (B, 4, 84, 80)
        Returns:
            torch.Tensor: Q-values for each action.
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
