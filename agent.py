import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.data import ReplayBuffer, LazyTensorStorage
import random

class NeuralNetwork(nn.Module):
    def __init__(self, n_obs: int, n_actions: int) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(n_obs, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)


class Agent:
    def __init__(self,
                 n_obs: int,
                 n_actions: int,
                 rb_len: int,
                 batch_size: int,
                 gamma: float,
                 lr: float,
                 ) -> None:
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma

        self.nn = NeuralNetwork(n_obs, n_actions)
        self.optim = torch.optim.Adam(self.nn.parameters(), lr=lr)

        self.eps = 0.1

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rb = ReplayBuffer(storage=LazyTensorStorage(rb_len, device=self.device))

    def select_action(self, obs: torch.Tensor) -> int:
        if random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            return self.nn.forward(obs).argmax().item()

    def update(self) -> None:
        if len(self.rb) < self.batch_size:
            return

        transitions = self.rb.sample(self.batch_size)

        obs = transitions['obs']
        actions = transitions['action'].long().unsqueeze(1)
        rewards = transitions['reward'].float()
        next_obs = transitions['next_obs']
        dones = transitions['is_done'].bool()

        q_preds = self.nn(obs).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q = self.nn(next_obs).max(dim=1).values
            q_targets = rewards + (~dones).float() * (self.gamma * next_q)

        loss = F.smooth_l1_loss(q_preds, q_targets)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.nn.parameters(), max_norm=10.0)
        self.optim.step()
