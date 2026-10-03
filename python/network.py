import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class AlphaZeroNet(nn.Module):
    def __init__(self, in_shape, num_actions, num_players, channels=64, blocks=5):
        super().__init__()
        planes, height, width = in_shape
        self.in_shape = tuple(in_shape)
        self.stem = nn.Sequential(
            nn.Conv2d(planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * height * width, num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(height * width, 64),
            nn.ReLU(),
            nn.Linear(64, num_players),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.tower(self.stem(x))
        return self.policy_head(x), self.value_head(x)


@torch.no_grad()
def evaluate(net, observations: np.ndarray, masks: np.ndarray, device):
    net.eval()
    x = torch.from_numpy(observations).to(device).view(-1, *net.in_shape)
    logits, values = net(x)
    legal = torch.from_numpy(masks).to(device)
    priors = torch.softmax(logits.masked_fill(~legal, float("-inf")), dim=1)
    return priors.float().cpu().numpy(), values.float().cpu().numpy()
