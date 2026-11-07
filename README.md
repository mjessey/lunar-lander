# Lunar Lander Workshop Series
This repo is for tracking the Lunar Lander Workshop series for the AI Club at TN Tech, a multi-day workshop series that covers some of the fundamentals of deep reinforcement learning (DRL). This workshop focuses on implementing the Dueling Double Deep Q-Network (D3QN) algorithm, with the purpose of training an agent to land a spaceship (lunar lander) at a specific location while not crashing and while minimizing fuel consumption.

## What is Deep Reinforcement Learning?
Reinforcement Learning (RL) is a subfield of Machine Learning that concerns itself with learning an optimal policy (strategy) for a given task (typically without the need for a model of the world's dynamics). In RL, an agent (the decision maker) interacts with an environment (the outside world) through an iterative and sequential process of taking actions in a particular state (determined by sensor readings, known as the observation) and then receiving feedback from the environment in the form of a scalar reward signal and a new state (or observation). The agent's objective is to learn an optimal mapping from states to action probabilities (known as a policy) such that sampling from this policy over many time steps would result in maximizing the cumulative (total) discounted future reward.

The agent learns to find the optimal policy by interacting with the environment through trial-and-error and then changing its current policy based on the feedback it receives. At every time step (a regular, discrete interval of time), the agent decides what action to take by predicting the expected sum of future rewards (the return) for each action. The agent selects the action with the highest expected return most of the time (this is called selecting greedily), but since there is uncertainty in its estimates, it may also select a random action some of the time so it can correct biases it may have and "explore" the state-space in hopes of finding a better policy.

Deep Reinforcement Learning (DRL) uses a deep neural network to approximate state-action values (represented by a function `Q(s, a)` where `s` is the current state and `a` is the action we are evaluating). To train a neural network, we must define a target so we can compute the loss and minimize it. Unfortunately, we do not know the true Q-values / expected returns (if we did we wouldn't need RL), so we must instead define a target that gives us an approximation of the true expected return that is usually better than our current predictions. We define this target as `r + γ V(s')` (this is called the temporal difference target) where `r` is the immediate reward, `γ` (Greek letter gamma) is the discount factor, and `V(s')` is the value of the next state `s'` (according to our neural network). In this workshop, we use Q-learning, which uses `V(s) = max_{for all a} Q(s, a)`, so the target is `r + γ max_{for all a} Q(s, a)`.

The Deep Q-Network (DQN) algorithm typically adds additional enhancements, like a replay buffer for decorrelating sampled states and improving sample efficiency and a dedicated target network for stabilizing the target. This workshop also adds additional optimizations, like double Q-learning, and dueling networks.

## Installation

### Step 1: Install Miniconda
Download and install Miniconda from https://www.anaconda.com/download/success?reg=skipped-miniconda
If you are using Linux, you should be able to install it instead through your distro's package manager

### Step 2: Set up Conda environment and install packages
In a terminal (Anaconda Prompt if using Windows)
```
conda create -n lunar-lander-env python=3.12
conda activate lunar-lander-env
python -m pip install --upgrade pip
pip install torch torchrl tensordict tensorboard "gym[box2d]"
```

### Step 3: Install Git (Windows Only, Linux comes with Git by default)
You can install git from https://git-scm.com/install/windows

### Step 4: Clone the repo
If you are on Windows, you will need to use the Git Bash terminal.
```
git clone https://github.com/mjessey/lunar-lander.git
cd lunar-lander
```

## Usage
From the repo root:
```
conda activate lunar-lander
python main.py
```
