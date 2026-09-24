import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class NeuralNetwork(nn.Module):
    def __init__(self, n_obs: int, n_actions: int) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(n_obs, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_3 = nn.Linear(64, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)


class Agent:
    def __init__(self, n_obs: int, n_actions: int) -> None:
        self.n_obs = n_obs
        self.n_actions = n_actions

        self.nn = NeuralNetwork(n_obs, n_actions)

        self.eps = 0.5

    def select_action(self, obs: torch.Tensor) -> int:
        if random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)

        return int(torch.argmax(self.nn.forward(obs)))