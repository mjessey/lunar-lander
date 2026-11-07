import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.data import ReplayBuffer, LazyTensorStorage
import random


class NeuralNetwork(nn.Module):
    """
    Neural network architecture:
    Input size: n_obs (8 in this case)
    Hidden layer 1 size: 128
    Hidden layer 2 size: 128
    Output size (layer 3): n_actions (4 in this case)
    """

    def __init__(self, n_obs: int, n_actions: int) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(n_obs, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass of the neural network (input -> output)
        Computes Q-values (state-action values)
        """
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)


class Agent:
    def __init__(self,
                 n_obs: int,        # size of observation vector
                 n_actions: int,    # size of action vector (number of actions)
                 rb_len: int,       # length of Replay Buffer (max transitions to be stored)
                 batch_size: int,   # number the transitions sampled at once from the replay buffer
                 gamma: float,      # discount factor (if <1, short-term rewards are weighted higher)
                 lr: float,         # learning rate (step size)
                 eps_decay: float   # used for decreasing epsilon to promote exploitation of knowledge
                 ) -> None:
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_decay = eps_decay

        # initializes neural network and optimizer (the thing training the neural network)
        self.nn = NeuralNetwork(n_obs, n_actions)
        self.optim = torch.optim.Adam(self.nn.parameters(), lr=lr)

        # epsilon is the probability of taking a random action (exploration)
        self.eps = 1.0

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # creates the Replay Buffer. Stores old transitions. Used for decorrelating transitions and
        # re-using old data improves sample efficiency (only used for off-policy learning)
        self.rb = ReplayBuffer(storage=LazyTensorStorage(rb_len, device=self.device))

    def select_action(self, obs: torch.Tensor) -> int:
        # selects a random action self.eps % of the time
        if random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)

        # selects the action with the highest expected future reward
        with torch.no_grad():
            return self.nn.forward(obs).argmax().item()

    def update(self) -> None:
        if len(self.rb) < self.batch_size:
            return

        # sampling (with replacement) self.batch_size transitions from the replay buffer
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
        torch.nn.utils.clip_grad_norm_(self.nn.parameters(), max_norm=10.0)
        self.optim.step()
