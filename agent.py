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
                 eps_decay: float,  # used for decreasing epsilon to promote exploitation of knowledge
                 seed_steps: int,   # number of transitions to seed the replay buffer before learning begins
                 sync_freq: int,  # how often we update the target network
                 ) -> None:
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_decay = eps_decay
        self.seed_steps = seed_steps
        self.sync_freq = sync_freq

        # initializes online network and optimizer (the thing training the neural network)
        self.online_net = NeuralNetwork(n_obs, n_actions)
        self.optim = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        # initializes target network (used for stabilizing training)
        self.target_net = NeuralNetwork(n_obs, n_actions)
        self.target_net.load_state_dict(self.online_net.state_dict())

        # epsilon is the probability of taking a random action (exploration)
        self.eps = 1.0

        # device to use for running the neural network and storing the replay buffer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # creates the Replay Buffer. Stores old transitions. Used for decorrelating transitions and
        # re-using old data improves sample efficiency (only used for off-policy learning)
        self.rb = ReplayBuffer(storage=LazyTensorStorage(rb_len, device=self.device))

        # total_number of training steps
        self.steps = 0

    def select_action(self, obs: torch.Tensor) -> int:
        # selects a random action self.eps % of the time
        if random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)

        # selects the action with the highest expected future reward
        with torch.no_grad():
            return self.online_net(obs).argmax().item()

    def update(self) -> None:
        # only train when the number of transitions in the replay buffer exceeds the seed steps
        if len(self.rb) < self.seed_steps:
            return

        # decay epsilon (promote exploitation)
        self.eps *= self.eps_decay

        # increment training steps
        self.steps += 1

        # sampling (with replacement) self.batch_size transitions from the replay buffer
        transitions = self.rb.sample(self.batch_size)

        # unpacking a batch from the replay buffer
        obs = transitions['obs']
        actions = transitions['action'].long().unsqueeze(1)
        rewards = transitions['reward'].float()
        next_obs = transitions['next_obs']
        dones = transitions['is_done'].bool()

        # gets the q predictions for the action actually taken in state obs
        q_preds = self.online_net(obs).gather(1, actions).squeeze(1)

        # computes the q targets according to double q learning
        # (reward + gamma * target_net_eval(best_action_according_to_online_net))
        with torch.no_grad():
            best_actions = self.online_net(next_obs).argmax(dim=1, keepdims=True)
            next_q = self.target_net(next_obs).gather(1, best_actions).squeeze(1)
            q_targets = rewards + (~dones).float() * (self.gamma * next_q)

        # computes the loss (Huber loss) or TD error
        loss = F.smooth_l1_loss(q_preds, q_targets)

        # update the online network
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optim.step()

        # update target network (sync it with online net)
        if self.steps % self.sync_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())