import gymnasium as gym
import torch
from tensordict import TensorDict

from agent import Agent

N_EPISODES = 1000

def run_episode(env: gym.Env, agent: Agent) -> float:
    obs, _ = env.reset()

    is_episode_done = False
    total_reward = 0
    while not is_episode_done:
        obs = torch.from_numpy(obs)
        action = agent.select_action(obs)
        next_obs, reward, is_terminated, is_truncated, _ = env.step(action)

        agent.rb.add(
            TensorDict({
                'obs': torch.as_tensor(obs, dtype=torch.float32, device=agent.device),
                'action': torch.as_tensor(action, dtype=torch.long, device=agent.device),
                'reward': torch.as_tensor(reward, dtype=torch.float32, device=agent.device),
                'next_obs': torch.as_tensor(next_obs, dtype=torch.float32, device=agent.device),
                'is_done': torch.as_tensor(is_terminated, dtype=torch.bool, device=agent.device),
            },
            batch_size=[],
        ))

        agent.update()

        obs = next_obs
        is_episode_done = is_terminated or is_truncated
        total_reward += reward

    return total_reward

def main() -> None:
    env = gym.make('LunarLander-v3')
    agent = Agent(n_obs=8,
                  n_actions=4,
                  rb_len=1e5,
                  batch_size=64,
                  gamma=0.99,
                  lr=1e-3,
                  )

    for episode in range(N_EPISODES):
        reward = run_episode(env, agent)
        print(f"Episode {episode + 1}\t Reward: {round(reward, 2)}\t Success: {reward>=200}")

if __name__ == '__main__':
    main()